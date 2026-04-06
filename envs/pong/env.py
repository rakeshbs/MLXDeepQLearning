import math
import os
import random

import numpy as np
import pygame

from envs.base import BaseEnv

SCREEN_WIDTH  = 640
SCREEN_HEIGHT = 480
FPS           = 60

PADDLE_W      = 12
PADDLE_H      = 80
PADDLE_SPEED  = 6   # pixels per step for both paddles

BALL_SIZE     = 10  # square half-size for collision; drawn as a rect of 2*BALL_SIZE
BALL_SPEED    = 5   # initial ball speed at each rally reset
MAX_BALL_SPEED = 12

# The left AI moves at this speed — kept below PADDLE_SPEED so the trained
# right agent has an exploitable edge while the AI is still challenging.
LEFT_AI_SPEED = 4

# Angle spread when ball hits a paddle (degrees).  A hit at the paddle edge
# returns the ball at MAX_ANGLE; a hit at the centre returns it nearly flat.
MAX_BOUNCE_ANGLE = 60.0

PIXEL_OBS_SIZE  = 84   # each frame is resized to 84×84
PIXEL_OBS_STACK = 4    # number of consecutive frames stacked into one observation


class PongEnv(BaseEnv):
    """
    Pong environment for reinforcement learning.

    Left paddle — simple AI (tracks ball with speed-capped pursuit).
    Right paddle — the learning agent.

    obs_type:
        "state"  — 6-dim float vector: (ball_x, ball_y, ball_vx, ball_vy,
                   right_cy, left_cy), all normalised. Fast; no pygame needed.
        "pixels" — 4 stacked 84×84 grayscale frames, shape (4, 84, 84).
                   Requires pygame even when render_mode=False (uses SDL dummy).

    Action space (right paddle only):
        0 — stay
        1 — move up   (decreasing y)
        2 — move down (increasing y)

    Reward:
        +1  when the right agent scores a point
        -1  when the left AI scores a point

    Episode terminates when either side reaches score_limit or max_steps.
    info["score"] is the number of points won by the right (agent) paddle.
    """

    obs_shape  = (6,)   # overridden to (4, 84, 84) in __init__ when obs_type="pixels"
    action_dim = 3

    def __init__(
        self,
        render_mode: bool = False,
        obs_type: str = "state",
        score_limit: int = 5,
        max_steps: int = 5_000,
    ):
        self.render_mode  = render_mode
        self.obs_type     = obs_type
        self.score_limit  = score_limit
        self.max_steps    = max_steps

        # Fixed paddle x positions (left edge of each paddle)
        self.left_px  = 20
        self.right_px = SCREEN_WIDTH - 20 - PADDLE_W

        needs_pygame = render_mode or obs_type == "pixels"
        if needs_pygame:
            if not render_mode:
                os.environ["SDL_VIDEODRIVER"] = "dummy"
            pygame.init()
            if render_mode:
                self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
                pygame.display.set_caption("Pong RL")
                self.font = pygame.font.SysFont(None, 48)
            else:
                pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
                self.screen = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT))
                self.font = None
            self.clock = pygame.time.Clock()
        else:
            self.screen = None
            self.font   = None

        if obs_type == "pixels":
            self.obs_shape   = (PIXEL_OBS_STACK, PIXEL_OBS_SIZE, PIXEL_OBS_SIZE)
            self._frame_stack = np.zeros(self.obs_shape, dtype=np.uint8)
        else:
            self.obs_shape = (6,)

        self.reset()

    # ------------------------------------------------------------------
    # BaseEnv interface
    # ------------------------------------------------------------------

    def reset(self) -> np.ndarray:
        self.left_cy   = SCREEN_HEIGHT / 2.0
        self.right_cy  = SCREEN_HEIGHT / 2.0
        self.left_score  = 0
        self.right_score = 0
        self.steps       = 0
        self._reset_ball(direction=random.choice([-1, 1]))
        if self.obs_type == "pixels":
            self._frame_stack = np.zeros(self.obs_shape, dtype=np.uint8)
            self._draw_frame()
            frame = self._capture_pixel_frame()
            for i in range(PIXEL_OBS_STACK):
                self._frame_stack[i] = frame
        return self._get_obs()

    def step(self, action: int) -> tuple[np.ndarray, float, bool, dict]:
        # --- right agent movement ---
        if action == 1:
            self.right_cy -= PADDLE_SPEED
        elif action == 2:
            self.right_cy += PADDLE_SPEED
        self.right_cy = float(np.clip(self.right_cy, PADDLE_H / 2, SCREEN_HEIGHT - PADDLE_H / 2))

        # --- left AI movement: capped-speed ball tracker ---
        diff = self.ball_y - self.left_cy
        move = min(abs(diff), LEFT_AI_SPEED) * (1 if diff > 0 else -1)
        self.left_cy = float(np.clip(self.left_cy + move, PADDLE_H / 2, SCREEN_HEIGHT - PADDLE_H / 2))

        # --- advance ball ---
        self.ball_x += self.ball_vx
        self.ball_y += self.ball_vy

        # --- top/bottom wall bounce ---
        if self.ball_y - BALL_SIZE <= 0:
            self.ball_y  = float(BALL_SIZE)
            self.ball_vy = abs(self.ball_vy)
        elif self.ball_y + BALL_SIZE >= SCREEN_HEIGHT:
            self.ball_y  = float(SCREEN_HEIGHT - BALL_SIZE)
            self.ball_vy = -abs(self.ball_vy)

        # --- paddle collisions ---
        self._check_left_paddle()
        self._check_right_paddle()

        # --- scoring ---
        reward = 0.0
        done   = False

        if self.ball_x - BALL_SIZE < 0:
            # Ball passed the left wall → right agent scores
            self.right_score += 1
            reward = 1.0
            if self.right_score >= self.score_limit:
                done = True
            else:
                self._reset_ball(direction=1)  # serve toward left after right scores

        elif self.ball_x + BALL_SIZE > SCREEN_WIDTH:
            # Ball passed the right wall → left AI scores
            self.left_score += 1
            reward = -1.0
            if self.left_score >= self.score_limit:
                done = True
            else:
                self._reset_ball(direction=-1)  # serve toward right after left scores

        self.steps += 1
        if self.steps >= self.max_steps:
            done = True

        if self.render_mode:
            self.render()
        elif self.obs_type == "pixels":
            self._draw_frame()

        if self.obs_type == "pixels":
            frame = self._capture_pixel_frame()
            self._frame_stack = np.roll(self._frame_stack, shift=-1, axis=0)
            self._frame_stack[-1] = frame

        return self._get_obs(), reward, done, {"score": self.right_score}

    def render(self) -> None:
        if not self.render_mode or self.screen is None:
            return
        self._draw_frame()
        pygame.display.flip()
        self.clock.tick(FPS)

    def capture_frame(self) -> "np.ndarray | None":
        if self.screen is None:
            return None
        raw = pygame.surfarray.array3d(self.screen)
        return raw.transpose(1, 0, 2)  # (W, H, 3) → (H, W, 3)

    def close(self) -> None:
        if self.render_mode:
            pygame.quit()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _reset_ball(self, direction: int = 1) -> None:
        """Place ball at centre and launch toward `direction` (+1 right, -1 left)."""
        self.ball_x  = SCREEN_WIDTH  / 2.0
        self.ball_y  = SCREEN_HEIGHT / 2.0
        angle        = random.uniform(-20.0, 20.0) * math.pi / 180.0
        self.ball_vx = float(direction * BALL_SPEED * math.cos(angle))
        self.ball_vy = float(BALL_SPEED * math.sin(angle))

    def _check_left_paddle(self) -> None:
        """Reflect ball off the left (AI) paddle if colliding."""
        left_edge  = self.left_px
        right_edge = self.left_px + PADDLE_W
        top_edge   = self.left_cy - PADDLE_H / 2
        bot_edge   = self.left_cy + PADDLE_H / 2

        if (self.ball_vx < 0
                and self.ball_x - BALL_SIZE <= right_edge
                and self.ball_x + BALL_SIZE >= left_edge
                and self.ball_y + BALL_SIZE >= top_edge
                and self.ball_y - BALL_SIZE <= bot_edge):
            # Push ball outside paddle to prevent tunnelling
            self.ball_x = right_edge + BALL_SIZE
            self._bounce(toward_right=True)

    def _check_right_paddle(self) -> None:
        """Reflect ball off the right (agent) paddle if colliding."""
        left_edge  = self.right_px
        right_edge = self.right_px + PADDLE_W
        top_edge   = self.right_cy - PADDLE_H / 2
        bot_edge   = self.right_cy + PADDLE_H / 2

        if (self.ball_vx > 0
                and self.ball_x + BALL_SIZE >= left_edge
                and self.ball_x - BALL_SIZE <= right_edge
                and self.ball_y + BALL_SIZE >= top_edge
                and self.ball_y - BALL_SIZE <= bot_edge):
            self.ball_x = left_edge - BALL_SIZE
            self._bounce(toward_right=False)

    def _bounce(self, toward_right: bool) -> None:
        """
        Reflect ball horizontally.

        The bounce angle is determined by where the ball hit the paddle
        relative to its centre: edge hit → steeper angle, centre hit → flatter.
        Speed increases by 5% per hit, capped at MAX_BALL_SPEED.
        """
        # Determine which paddle centre to use for offset calculation
        cy     = self.right_cy if not toward_right else self.left_cy
        offset = (self.ball_y - cy) / (PADDLE_H / 2.0)            # [-1, 1]
        offset = max(-1.0, min(1.0, offset))
        angle  = offset * MAX_BOUNCE_ANGLE * math.pi / 180.0

        current_speed = math.hypot(self.ball_vx, self.ball_vy)
        new_speed     = min(current_speed * 1.05, MAX_BALL_SPEED)

        direction  = 1 if toward_right else -1
        self.ball_vx = float(direction * new_speed * math.cos(angle))
        self.ball_vy = float(new_speed * math.sin(angle))

    def _capture_pixel_frame(self) -> np.ndarray:
        """
        Capture the current pygame surface as an 84×84 uint8 grayscale frame.

        surfarray.array3d returns (W, H, 3); transpose to (H, W, 3) then
        convert to grayscale using BT.601 luminance coefficients.
        Nearest-neighbour resize keeps it fast and avoids interpolation blur.
        """
        raw  = pygame.surfarray.array3d(self.screen)  # (W, H, 3)
        raw  = np.transpose(raw, (1, 0, 2))            # (H, W, 3)
        gray = (0.299 * raw[:, :, 0] + 0.587 * raw[:, :, 1] + 0.114 * raw[:, :, 2])
        h, w    = gray.shape
        th = tw = PIXEL_OBS_SIZE
        row_idx = (np.arange(th) * h // th).astype(np.int32)
        col_idx = (np.arange(tw) * w // tw).astype(np.int32)
        return gray[np.ix_(row_idx, col_idx)].astype(np.uint8)

    def _get_obs(self) -> np.ndarray:
        if self.obs_type == "pixels":
            return self._frame_stack.copy()
        return np.array([
            self.ball_x  / SCREEN_WIDTH,
            self.ball_y  / SCREEN_HEIGHT,
            self.ball_vx / MAX_BALL_SPEED,
            self.ball_vy / MAX_BALL_SPEED,
            self.right_cy / SCREEN_HEIGHT,
            self.left_cy  / SCREEN_HEIGHT,
        ], dtype=np.float32)

    def _draw_frame(self) -> None:
        pygame.event.pump()
        self.screen.fill((0, 0, 0))

        # Centre net (dashed)
        cx = SCREEN_WIDTH // 2
        dash = 10
        for y in range(0, SCREEN_HEIGHT, dash * 2):
            pygame.draw.rect(self.screen, (80, 80, 80), (cx - 1, y, 2, dash))

        # Left paddle (AI)
        lrect = pygame.Rect(
            self.left_px,
            int(self.left_cy - PADDLE_H / 2),
            PADDLE_W, PADDLE_H,
        )
        pygame.draw.rect(self.screen, (200, 200, 200), lrect)

        # Right paddle (agent)
        rrect = pygame.Rect(
            self.right_px,
            int(self.right_cy - PADDLE_H / 2),
            PADDLE_W, PADDLE_H,
        )
        pygame.draw.rect(self.screen, (200, 200, 200), rrect)

        # Ball
        brect = pygame.Rect(
            int(self.ball_x - BALL_SIZE),
            int(self.ball_y - BALL_SIZE),
            BALL_SIZE * 2, BALL_SIZE * 2,
        )
        pygame.draw.rect(self.screen, (255, 255, 255), brect)

        # Scores (only drawn in render_mode where a font is available)
        if self.font is not None:
            left_text  = self.font.render(str(self.left_score),  True, (180, 180, 180))
            right_text = self.font.render(str(self.right_score), True, (255, 255, 255))
            self.screen.blit(left_text,  (SCREEN_WIDTH // 4 - left_text.get_width() // 2,  20))
            self.screen.blit(right_text, (3 * SCREEN_WIDTH // 4 - right_text.get_width() // 2, 20))
