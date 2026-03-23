import os
import random

import numpy as np
import pygame

from envs.base import BaseEnv

SCREEN_WIDTH    = 512
SCREEN_HEIGHT   = 512
FPS             = 30
PIXEL_OBS_SIZE  = 84   # each frame is resized to 84×84
PIXEL_OBS_STACK = 4    # number of consecutive frames stacked into one observation


class FlappyBirdEnv(BaseEnv):
    """
    Flappy Bird environment for reinforcement learning using actual sprites.
    Provides a Gym-like interface: reset(), step(), render().

    obs_type:
        "state"  — 5-dim float vector: (bird_y, velocity, pipe_dist,
                   top_pipe_y_relative, bottom_pipe_y_relative). Fast and
                   suitable for MLP networks; pygame is not needed.
        "pixels" — 4 stacked 84×84 grayscale frames, shape (4, 84, 84).
                   Requires pygame even when not rendering visually. When
                   render_mode is False, SDL_VIDEODRIVER=dummy is set so
                   pygame renders to a surface without a real display server.

    action space: 0 = do nothing, 1 = flap (sets upward velocity)
    """

    action_dim = 2

    def __init__(self, render_mode=False, obs_type="state"):
        self.render_mode  = render_mode
        self.obs_type     = obs_type
        self.screen_width  = SCREEN_WIDTH
        self.screen_height = SCREEN_HEIGHT

        # Physics constants
        self.gravity      = 1
        self.flap_power   = -9    # negative = upward in pygame's y-down coordinate system
        self.max_velocity = 10    # terminal velocity cap prevents instant-death dives
        self.pipe_gap     = 100   # vertical gap the bird must fly through
        self.pipe_velocity = -4   # pipes move left at this many pixels per step

        needs_pygame = render_mode or obs_type == "pixels"
        if needs_pygame:
            if not render_mode:
                # SDL_VIDEODRIVER=dummy tells SDL to use a null display driver so
                # pygame.init() and display.set_mode() succeed without a real screen.
                # This is required for pixel observations in headless actor processes.
                os.environ["SDL_VIDEODRIVER"] = "dummy"
            pygame.init()
            if render_mode:
                self.screen = pygame.display.set_mode(
                    (self.screen_width, self.screen_height)
                )
                pygame.display.set_caption("Flappy Bird RL")
            else:
                # Even in offscreen mode, set_mode() must be called so that
                # Surface.convert() has a pixel format to convert to. Without
                # this call, convert() raises an error about no video mode.
                pygame.display.set_mode((self.screen_width, self.screen_height))
                self.screen = pygame.Surface((self.screen_width, self.screen_height))
            self.clock = pygame.time.Clock()
            self._load_assets()
        else:
            # For state observations without rendering, hard-code asset dimensions
            # so the physics and state normalisation still work correctly.
            self.bird_width  = 34
            self.bird_height = 24
            self.pipe_width  = 52
            self.pipe_height = 320
            self.base_height = 112

        if obs_type == "pixels":
            # Channel-first layout (C, H, W) matches the CNN input convention
            self.obs_shape   = (PIXEL_OBS_STACK, PIXEL_OBS_SIZE, PIXEL_OBS_SIZE)
            self._frame_stack = np.zeros(self.obs_shape, dtype=np.float32)
        else:
            self.obs_shape = (5,)

        self.pipes        = []
        self.score        = 0
        self.steps        = 0

        self.bird_x        = int(self.screen_width * 0.2)  # bird stays at fixed x; pipes scroll
        self.bird_y        = int((self.screen_height - getattr(self, "base_height", 112)) / 2)
        self.bird_velocity = 0
        self.base_x        = 0   # scrolling offset for the ground animation

        self.reset()

    def _load_assets(self):
        """Load and cache sprite images from the assets subdirectory."""
        assets_dir = os.path.join(os.path.dirname(__file__), "assets")

        # convert() / convert_alpha() converts to the display pixel format for faster blitting
        self.bg_img = pygame.image.load(
            os.path.join(assets_dir, "background-day.png")
        ).convert()
        self.base_img = pygame.image.load(
            os.path.join(assets_dir, "base.png")
        ).convert_alpha()

        # Three animation frames: downflap → midflap → upflap
        self.bird_imgs = [
            pygame.image.load(
                os.path.join(assets_dir, "yellowbird-downflap.png")
            ).convert_alpha(),
            pygame.image.load(
                os.path.join(assets_dir, "yellowbird-midflap.png")
            ).convert_alpha(),
            pygame.image.load(
                os.path.join(assets_dir, "yellowbird-upflap.png")
            ).convert_alpha(),
        ]

        pipe_img = pygame.image.load(
            os.path.join(assets_dir, "pipe-green.png")
        ).convert_alpha()
        self.pipe_bottom_img = pipe_img
        # Top pipe is the bottom pipe flipped vertically
        self.pipe_top_img = pygame.transform.flip(pipe_img, False, True)

        # Cache dimensions so physics code doesn't call get_width/height every step
        self.bird_width  = self.bird_imgs[0].get_width()
        self.bird_height = self.bird_imgs[0].get_height()
        self.pipe_width  = pipe_img.get_width()
        self.pipe_height = pipe_img.get_height()
        self.base_height = self.base_img.get_height()

    def reset(self) -> np.ndarray:
        """Reset environment to start state and return the first observation."""
        self.bird_y        = int((self.screen_height - self.base_height) / 2)
        self.bird_velocity = 0
        self.pipes         = []
        self._spawn_pipe()   # always start with one pipe visible
        self.score         = 0
        self.steps         = 0
        self.bird_anim_idx = 0
        self.base_x        = 0
        if self.obs_type == "pixels":
            self._frame_stack = np.zeros(self.obs_shape, dtype=np.float32)
            # Draw the initial frame once so the surface is populated, then
            # fill all 4 stack slots with the same frame. This avoids a
            # misleading all-black first observation for the first 3 steps.
            self._draw_frame()
            frame = self._capture_frame()
            for i in range(PIXEL_OBS_STACK):
                self._frame_stack[i] = frame
        return self._get_obs()

    def _spawn_pipe(self):
        """
        Spawn a new pipe pair at the right edge of the screen.

        The gap top (pipe_top_y) is drawn uniformly from the middle 60% of the
        playable height to ensure the gap is always reachable.
        """
        usable_height = self.screen_height - self.base_height
        min_y = int(usable_height * 0.2)
        max_y = int(usable_height * 0.8) - self.pipe_gap
        pipe_top_y = random.randint(min_y, max_y)
        self.pipes.append(
            {
                "x":      self.screen_width,          # spawn off-screen right
                "top":    pipe_top_y,                 # y-coord of gap top
                "bottom": pipe_top_y + self.pipe_gap, # y-coord of gap bottom
                "passed": False,                      # True once the bird clears this pipe
            }
        )

    def step(self, action: int) -> tuple[np.ndarray, float, bool, dict]:
        """
        Advance the game by one time step.

        Physics update order: flap → gravity → velocity cap → position.
        Pipe collision and scoring are checked after position is updated.
        """
        if action == 1:
            # Flap overrides any downward velocity instantly (no momentum carry-over)
            self.bird_velocity = self.flap_power

        self.bird_velocity += self.gravity
        if self.bird_velocity > self.max_velocity:
            self.bird_velocity = self.max_velocity  # cap to prevent unlimited acceleration
        self.bird_y += self.bird_velocity

        self.steps += 1
        reward = 0.1   # small positive reward for surviving each step
        done   = False

        # Scroll the ground texture; wrap at -336 (width of base sprite tile)
        self.base_x += self.pipe_velocity
        if self.base_x <= -336:
            self.base_x += 336

        # Move all pipes leftward by pipe_velocity (negative = left)
        for pipe in self.pipes:
            pipe["x"] += self.pipe_velocity

        # Remove pipes that have scrolled far off the left edge
        if self.pipes and self.pipes[0]["x"] < -self.pipe_width * 2:
            self.pipes.pop(0)

        # Spawn a new pipe when the last one has scrolled far enough left
        if self.pipes and self.pipes[-1]["x"] < self.screen_width - 150:
            self._spawn_pipe()

        # Shrink the bird's collision rect by 4 px on each side for leniency —
        # pixel-perfect collision on sprites with transparent edges feels unfair.
        bird_rect = pygame.Rect(
            self.bird_x + 4, self.bird_y + 4, self.bird_width - 8, self.bird_height - 8
        )

        # Check boundary collisions (ground and ceiling)
        if (
            self.bird_y + self.bird_height >= self.screen_height - self.base_height
            or self.bird_y < 0
        ):
            done = True

        for pipe in self.pipes:
            # Top pipe extends upward from pipe["top"] by pipe_height pixels
            top_pipe_rect    = pygame.Rect(
                pipe["x"], pipe["top"] - self.pipe_height, self.pipe_width, self.pipe_height
            )
            # Bottom pipe extends downward from pipe["bottom"]
            bottom_pipe_rect = pygame.Rect(
                pipe["x"], pipe["bottom"], self.pipe_width, self.pipe_height
            )
            if bird_rect.colliderect(top_pipe_rect) or bird_rect.colliderect(bottom_pipe_rect):
                done = True

            # Score a point when the pipe's right edge passes the bird's left edge
            if not pipe["passed"] and pipe["x"] + self.pipe_width < self.bird_x:
                pipe["passed"] = True
                self.score    += 1
                reward         = 10

        # In render_mode, draw to screen and flip display; in pixel-obs mode, draw offscreen
        if self.render_mode:
            self.render()
        elif self.obs_type == "pixels":
            self._draw_frame()

        if self.obs_type == "pixels":
            frame = self._capture_frame()
            # Roll the stack by -1 along axis 0 (channel/time axis) to shift
            # old frames forward, then overwrite the last slot with the new frame.
            self._frame_stack = np.roll(self._frame_stack, shift=-1, axis=0)
            self._frame_stack[-1] = frame

        return self._get_obs(), reward, done, {"score": self.score}

    def _get_obs(self) -> np.ndarray:
        """Return the observation appropriate for the configured obs_type."""
        if self.obs_type == "pixels":
            return self._frame_stack.copy()  # copy so the caller can't corrupt the internal stack
        return self._get_state()

    def _capture_frame(self) -> np.ndarray:
        """
        Capture the current pygame surface and return an 84×84 grayscale frame in [0, 1].

        surfarray.array3d returns a (W, H, 3) array in pygame's x-major layout;
        transposing to (H, W, 3) converts to standard image row-major layout before
        the grayscale conversion. Resizing uses nearest-neighbour (integer index
        sampling) rather than bilinear interpolation — fast enough and sufficient
        for the pixel-observation task.
        """
        raw  = pygame.surfarray.array3d(self.screen)  # (W, H, 3), uint8
        raw  = np.transpose(raw, (1, 0, 2))            # (H, W, 3)
        # Luminance conversion (ITU-R BT.601 coefficients)
        gray = (0.299 * raw[:, :, 0] + 0.587 * raw[:, :, 1] + 0.114 * raw[:, :, 2])
        # Nearest-neighbour resize using integer index sampling (no interpolation)
        h, w    = gray.shape
        th = tw = PIXEL_OBS_SIZE
        row_idx = (np.arange(th) * h // th).astype(np.int32)
        col_idx = (np.arange(tw) * w // tw).astype(np.int32)
        resized = gray[np.ix_(row_idx, col_idx)]
        return (resized / 255.0).astype(np.float32)  # normalise to [0, 1]

    def _get_state(self) -> np.ndarray:
        """
        Build the 5-dimensional state vector, normalised to roughly [-1, 1].

        All features are divided by their natural scale (screen dimensions,
        max velocity) so the network receives inputs of similar magnitude,
        which speeds up learning and improves numerical stability.

        Features:
          [0] bird_y / screen_height               — vertical position
          [1] bird_velocity / max_velocity          — current speed
          [2] dist_to_pipe / screen_width           — horizontal distance to next pipe
          [3] (top_pipe_y - bird_y) / screen_height — distance above gap top (signed)
          [4] (bottom_pipe_y - bird_y) / screen_height — distance below gap bottom (signed)
        """
        # Find the first pipe whose right edge is still ahead of the bird
        next_pipe = None
        for pipe in self.pipes:
            if pipe["x"] + self.pipe_width > self.bird_x:
                next_pipe = pipe
                break

        if next_pipe is None:
            # No pipe visible yet; use safe fallback values
            dist_to_pipe   = self.screen_width
            top_pipe_y     = 0
            bottom_pipe_y  = self.screen_height - self.base_height
        else:
            dist_to_pipe   = next_pipe["x"] - self.bird_x
            top_pipe_y     = next_pipe["top"]
            bottom_pipe_y  = next_pipe["bottom"]

        state = [
            self.bird_y / self.screen_height,
            self.bird_velocity / self.max_velocity,
            dist_to_pipe / self.screen_width,
            (top_pipe_y - self.bird_y) / self.screen_height,
            (bottom_pipe_y - self.bird_y) / self.screen_height,
        ]
        return np.array(state, dtype=np.float32)

    def _draw_frame(self) -> None:
        """
        Draw the current game state onto self.screen (works offscreen too).

        pygame.event.pump() must be called even in headless mode to prevent
        the SDL event queue from filling up, which would cause pygame to hang.
        """
        pygame.event.pump()

        # Tile the background image horizontally to fill the screen width
        bg_w = self.bg_img.get_width()
        for x in range(0, self.screen_width, bg_w):
            self.screen.blit(self.bg_img, (x, 0))

        # Draw pipe pairs: top pipe hangs down from pipe["top"], bottom pipe rises from pipe["bottom"]
        for pipe in self.pipes:
            self.screen.blit(self.pipe_top_img,    (pipe["x"], pipe["top"] - self.pipe_height))
            self.screen.blit(self.pipe_bottom_img, (pipe["x"], pipe["bottom"]))

        # Scroll the ground by tiling the base image starting at base_x (wraps around)
        base_w = self.base_img.get_width()
        base_y = self.screen_height - self.base_height
        x      = self.base_x
        while x < self.screen_width:
            self.screen.blit(self.base_img, (x, base_y))
            x += base_w

        # Advance bird animation frame every 5 steps (cycles through 3 flap images)
        if self.steps % 5 == 0:
            self.bird_anim_idx = (self.bird_anim_idx + 1) % 3

        bird_img = self.bird_imgs[self.bird_anim_idx]
        # Tilt the sprite proportional to velocity: positive velocity (falling) → nose down
        rotation = -min(self.bird_velocity * 3, 90)
        rotated_bird = pygame.transform.rotate(bird_img, rotation)
        # Keep the rotated image centred on the bird's logical position
        rect = rotated_bird.get_rect(
            center=(self.bird_x + self.bird_width // 2, self.bird_y + self.bird_height // 2)
        )
        self.screen.blit(rotated_bird, rect)

        font       = pygame.font.SysFont(None, 48)
        score_text = font.render(str(self.score), True, (255, 255, 255))
        self.screen.blit(score_text, (self.screen_width // 2 - score_text.get_width() // 2, 50))

    def render(self) -> None:
        """Render to the real display and advance the frame clock to FPS."""
        if not self.render_mode:
            return
        self._draw_frame()
        pygame.display.flip()
        self.clock.tick(FPS)

    def close(self) -> None:
        """Release pygame resources. Only called in render_mode to avoid quitting a headless session."""
        if self.render_mode:
            pygame.quit()
