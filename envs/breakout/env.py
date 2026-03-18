import os
import random

import numpy as np
import pygame

from envs.base import BaseEnv

SCREEN_WIDTH = 512
SCREEN_HEIGHT = 512
FPS = 60
PIXEL_OBS_SIZE = 84
PIXEL_OBS_STACK = 4

PADDLE_WIDTH = 72
PADDLE_HEIGHT = 14
PADDLE_Y = SCREEN_HEIGHT - 36
PADDLE_SPEED = 8

BALL_SIZE = 10
BALL_SPEED = 5.0
BALL_SPEEDUP = 1.015
MAX_BALL_SPEED = 8.0
MAX_STEPS = 20_000  # safety cap; not part of the original Atari rules

BRICK_ROWS = 6
BRICK_COLS = 18
BRICK_HEIGHT = 18
BRICK_TOP = 58
BRICK_VALUES = [7, 7, 4, 4, 1, 1]
BRICK_COLORS = [
    (220, 72, 72),   # red
    (242, 127, 42),  # orange
    (241, 196, 15),  # yellow
    (90, 190, 90),   # green
    (79, 195, 247),  # aqua
    (66, 133, 244),  # blue
]

MAX_LIVES = 5
MAX_WALLS = 2
MAX_SCORE = BRICK_COLS * sum(BRICK_VALUES) * MAX_WALLS  # 864

ACTION_NOOP = 0
ACTION_FIRE = 1
ACTION_RIGHT = 2
ACTION_LEFT = 3


class BreakoutEnv(BaseEnv):
    """
    Atari-inspired Breakout environment using the original scoring structure.

    Modeled after standard one-player Atari Breakout rules:
      - 5 lives
      - 2 walls total per game
      - reward equals score delta from bricks only
      - actions: NOOP, FIRE, RIGHT, LEFT

    This is not a ROM-accurate emulator, but the rules and point system track
    the Atari manual and ALE Breakout documentation closely.
    """

    action_dim = 4

    def __init__(self, render_mode: bool = False, obs_type: str = "state"):
        self.render_mode = render_mode
        self.obs_type = obs_type
        self.screen_width = SCREEN_WIDTH
        self.screen_height = SCREEN_HEIGHT

        needs_pygame = render_mode or obs_type == "pixels"
        self._pygame_active = needs_pygame
        if needs_pygame:
            if not render_mode:
                os.environ["SDL_VIDEODRIVER"] = "dummy"
            pygame.init()
            if render_mode:
                self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
                pygame.display.set_caption("Breakout RL")
            else:
                pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
                self.screen = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT))
            self.clock = pygame.time.Clock()
        else:
            self.screen = None

        if obs_type == "pixels":
            self.obs_shape = (PIXEL_OBS_STACK, PIXEL_OBS_SIZE, PIXEL_OBS_SIZE)
            self._frame_stack = np.zeros(self.obs_shape, dtype=np.float32)
        else:
            self.obs_shape = (8,)

        self.paddle_x = 0.0
        self.ball_x = 0.0
        self.ball_y = 0.0
        self.ball_vx = 0.0
        self.ball_vy = 0.0
        self.ball_attached = True
        self.bricks = []
        self.score = 0
        self.steps = 0
        self.lives = MAX_LIVES
        self.walls_cleared = 0

        self.reset()

    def reset(self) -> np.ndarray:
        self.paddle_x = (SCREEN_WIDTH - PADDLE_WIDTH) / 2
        self.score = 0
        self.steps = 0
        self.lives = MAX_LIVES
        self.walls_cleared = 0
        self.bricks = self._build_wall()
        self._attach_ball()

        if self.obs_type == "pixels":
            self._frame_stack = np.zeros(self.obs_shape, dtype=np.float32)
            self._draw_frame()
            frame = self._capture_frame()
            for i in range(PIXEL_OBS_STACK):
                self._frame_stack[i] = frame

        return self._get_obs()

    def step(self, action: int) -> tuple[np.ndarray, float, bool, dict]:
        self.steps += 1

        if action == ACTION_LEFT:
            self.paddle_x -= PADDLE_SPEED
        elif action == ACTION_RIGHT:
            self.paddle_x += PADDLE_SPEED
        self.paddle_x = float(np.clip(self.paddle_x, 0, SCREEN_WIDTH - PADDLE_WIDTH))

        reward = 0.0
        done = False

        if self.ball_attached:
            self._attach_ball()
            if action == ACTION_FIRE:
                self._launch_ball()
        else:
            reward, done = self._advance_ball()

        if self.steps >= MAX_STEPS:
            done = True

        if self.render_mode:
            self.render()
        elif self.obs_type == "pixels":
            self._draw_frame()

        if self.obs_type == "pixels":
            frame = self._capture_frame()
            self._frame_stack = np.roll(self._frame_stack, shift=-1, axis=0)
            self._frame_stack[-1] = frame

        info = {
            "score": self.score,
            "lives": self.lives,
            "walls_cleared": self.walls_cleared,
            "max_score": MAX_SCORE,
        }
        return self._get_obs(), reward, done, info

    def _advance_ball(self) -> tuple[float, bool]:
        reward = 0.0
        done = False

        substeps = max(
            1, int(np.ceil(max(abs(self.ball_vx), abs(self.ball_vy)) / BALL_SIZE))
        )
        sub_vx = self.ball_vx / substeps
        sub_vy = self.ball_vy / substeps

        for _ in range(substeps):
            prev_x = self.ball_x
            prev_y = self.ball_y
            self.ball_x += sub_vx
            self.ball_y += sub_vy

            if self.ball_x <= 0:
                self.ball_x = 0
                self.ball_vx = abs(self.ball_vx)
                sub_vx = abs(sub_vx)
            elif self.ball_x + BALL_SIZE >= SCREEN_WIDTH:
                self.ball_x = SCREEN_WIDTH - BALL_SIZE
                self.ball_vx = -abs(self.ball_vx)
                sub_vx = -abs(sub_vx)

            if self.ball_y <= 0:
                self.ball_y = 0
                self.ball_vy = abs(self.ball_vy)
                sub_vy = abs(sub_vy)

            paddle_rect = pygame.Rect(int(self.paddle_x), PADDLE_Y, PADDLE_WIDTH, PADDLE_HEIGHT)
            ball_rect = pygame.Rect(int(self.ball_x), int(self.ball_y), BALL_SIZE, BALL_SIZE)

            if ball_rect.colliderect(paddle_rect) and self.ball_vy > 0:
                self.ball_y = PADDLE_Y - BALL_SIZE - 1
                self.ball_vy = -abs(self.ball_vy)
                hit_pos = (
                    (self.ball_x + BALL_SIZE / 2) - (self.paddle_x + PADDLE_WIDTH / 2)
                ) / (PADDLE_WIDTH / 2)
                hit_pos = float(np.clip(hit_pos, -1.0, 1.0))
                self.ball_vx = hit_pos * (MAX_BALL_SPEED * 0.8)
                self._speed_up_ball()
                self._clamp_ball_speed(min_vertical_speed=3.0)
                sub_vx = self.ball_vx / substeps
                sub_vy = self.ball_vy / substeps
                ball_rect = pygame.Rect(int(self.ball_x), int(self.ball_y), BALL_SIZE, BALL_SIZE)

            brick_hit = None
            for brick in self.bricks:
                brick_rect = brick["rect"]
                if ball_rect.colliderect(brick_rect):
                    brick_hit = brick
                    overlap_left = ball_rect.right - brick_rect.left
                    overlap_right = brick_rect.right - ball_rect.left
                    overlap_top = ball_rect.bottom - brick_rect.top
                    overlap_bottom = brick_rect.bottom - ball_rect.top
                    min_overlap = min(
                        overlap_left, overlap_right, overlap_top, overlap_bottom
                    )

                    if min_overlap in (overlap_left, overlap_right):
                        self.ball_vx = -self.ball_vx
                        if overlap_left < overlap_right:
                            self.ball_x = brick_rect.left - BALL_SIZE - 1
                        else:
                            self.ball_x = brick_rect.right + 1
                        sub_vx = self.ball_vx / substeps
                    else:
                        self.ball_vy = -self.ball_vy
                        if overlap_top < overlap_bottom:
                            self.ball_y = brick_rect.top - BALL_SIZE - 1
                        else:
                            self.ball_y = brick_rect.bottom + 1
                        sub_vy = self.ball_vy / substeps
                    break

            if brick_hit is not None:
                self.bricks.remove(brick_hit)
                self.score += brick_hit["value"]
                reward += float(brick_hit["value"])
                self._speed_up_ball()
                self._clamp_ball_speed(min_vertical_speed=2.5)
                sub_vx = self.ball_vx / substeps
                sub_vy = self.ball_vy / substeps

                if not self.bricks:
                    self.walls_cleared += 1
                    if self.walls_cleared >= MAX_WALLS:
                        done = True
                    else:
                        self.bricks = self._build_wall()
                    break

            if self.ball_y > SCREEN_HEIGHT:
                self.lives -= 1
                if self.lives <= 0:
                    done = True
                else:
                    self._attach_ball()
                break

        return reward, done

    def _build_wall(self) -> list[dict]:
        bricks = []
        for row, value in enumerate(BRICK_VALUES):
            color = BRICK_COLORS[row]
            top = BRICK_TOP + row * BRICK_HEIGHT
            bottom = top + BRICK_HEIGHT
            for col in range(BRICK_COLS):
                left = round(col * SCREEN_WIDTH / BRICK_COLS)
                right = round((col + 1) * SCREEN_WIDTH / BRICK_COLS)
                bricks.append(
                    {
                        "rect": pygame.Rect(left, top, right - left, bottom - top),
                        "color": color,
                        "value": value,
                    }
                )
        return bricks

    def _attach_ball(self) -> None:
        self.ball_attached = True
        self.ball_x = self.paddle_x + PADDLE_WIDTH / 2 - BALL_SIZE / 2
        self.ball_y = PADDLE_Y - BALL_SIZE - 4
        self.ball_vx = 0.0
        self.ball_vy = 0.0

    def _launch_ball(self) -> None:
        self.ball_attached = False
        angle = random.uniform(28, 55) * random.choice([-1, 1])
        rad = np.deg2rad(angle)
        self.ball_vx = BALL_SPEED * np.sin(rad)
        self.ball_vy = -BALL_SPEED * np.cos(rad)

    def _clamp_ball_speed(self, min_vertical_speed: float = 0.0) -> None:
        speed = np.sqrt(self.ball_vx**2 + self.ball_vy**2)
        if speed > MAX_BALL_SPEED:
            scale = MAX_BALL_SPEED / speed
            self.ball_vx *= scale
            self.ball_vy *= scale

        if min_vertical_speed > 0 and abs(self.ball_vy) < min_vertical_speed:
            self.ball_vy = np.sign(self.ball_vy or -1.0) * min_vertical_speed
            speed = np.sqrt(self.ball_vx**2 + self.ball_vy**2)
            if speed > MAX_BALL_SPEED:
                scale = MAX_BALL_SPEED / speed
                self.ball_vx *= scale
                self.ball_vy *= scale

    def _speed_up_ball(self) -> None:
        speed = np.sqrt(self.ball_vx**2 + self.ball_vy**2)
        if speed == 0:
            return
        new_speed = min(MAX_BALL_SPEED, speed * BALL_SPEEDUP)
        scale = new_speed / speed
        self.ball_vx *= scale
        self.ball_vy *= scale

    def _get_obs(self) -> np.ndarray:
        if self.obs_type == "pixels":
            return self._frame_stack.copy()
        return self._get_state()

    def _get_state(self) -> np.ndarray:
        return np.array(
            [
                (self.ball_x + BALL_SIZE / 2) / SCREEN_WIDTH,
                (self.ball_y + BALL_SIZE / 2) / SCREEN_HEIGHT,
                self.ball_vx / MAX_BALL_SPEED,
                self.ball_vy / MAX_BALL_SPEED,
                (self.paddle_x + PADDLE_WIDTH / 2) / SCREEN_WIDTH,
                len(self.bricks) / (BRICK_ROWS * BRICK_COLS),
                self.lives / MAX_LIVES,
                float(self.ball_attached),
            ],
            dtype=np.float32,
        )

    def _draw_frame(self) -> None:
        if self.screen is None:
            return
        pygame.event.pump()

        self.screen.fill((8, 12, 22))

        for brick in self.bricks:
            pygame.draw.rect(self.screen, brick["color"], brick["rect"])

        pygame.draw.rect(
            self.screen,
            (240, 240, 240),
            (int(self.paddle_x), PADDLE_Y, PADDLE_WIDTH, PADDLE_HEIGHT),
            border_radius=4,
        )
        pygame.draw.rect(
            self.screen,
            (255, 255, 255),
            (int(self.ball_x), int(self.ball_y), BALL_SIZE, BALL_SIZE),
        )

        font = pygame.font.SysFont(None, 28)
        score_text = font.render(f"Score: {self.score}", True, (235, 235, 235))
        lives_text = font.render(f"Lives: {self.lives}", True, (210, 220, 235))
        walls_text = font.render(f"Walls: {self.walls_cleared}/{MAX_WALLS}", True, (210, 220, 235))
        self.screen.blit(score_text, (10, 14))
        self.screen.blit(lives_text, (SCREEN_WIDTH // 2 - lives_text.get_width() // 2, 14))
        self.screen.blit(walls_text, (SCREEN_WIDTH - walls_text.get_width() - 10, 14))

        if self.ball_attached:
            serve_text = font.render("Press Space to Serve", True, (255, 220, 130))
            self.screen.blit(
                serve_text,
                (SCREEN_WIDTH // 2 - serve_text.get_width() // 2, PADDLE_Y - 42),
            )

    def _capture_frame(self) -> np.ndarray:
        raw = pygame.surfarray.array3d(self.screen)
        raw = np.transpose(raw, (1, 0, 2))
        gray = 0.299 * raw[:, :, 0] + 0.587 * raw[:, :, 1] + 0.114 * raw[:, :, 2]
        h, w = gray.shape
        th = tw = PIXEL_OBS_SIZE
        row_idx = (np.arange(th) * h // th).astype(np.int32)
        col_idx = (np.arange(tw) * w // tw).astype(np.int32)
        resized = gray[np.ix_(row_idx, col_idx)]
        return (resized / 255.0).astype(np.float32)

    def render(self) -> None:
        if not self.render_mode:
            return
        self._draw_frame()
        pygame.display.flip()
        self.clock.tick(FPS)

    def close(self) -> None:
        if self._pygame_active:
            pygame.quit()
