"""Play one of the RL environments manually."""

import argparse

import pygame

from envs.breakout import BreakoutEnv
from envs.flappy_bird import FlappyBirdEnv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Play an environment manually.")
    parser.add_argument(
        "--game",
        choices=("breakout", "flappy"),
        default="breakout",
        help="Which game to launch.",
    )
    return parser.parse_args()


def _process_events() -> tuple[bool, bool, bool]:
    running = True
    trigger = False
    reset = False

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.KEYDOWN:
            if event.key in (pygame.K_ESCAPE, pygame.K_q):
                running = False
            elif event.key == pygame.K_SPACE:
                trigger = True
            elif event.key == pygame.K_r:
                reset = True

    return running, trigger, reset


def _breakout_action() -> int:
    keys = pygame.key.get_pressed()
    left = keys[pygame.K_a] or keys[pygame.K_LEFT]
    right = keys[pygame.K_d] or keys[pygame.K_RIGHT]

    if right and not left:
        return 2
    if left and not right:
        return 3
    return 0


def play_breakout() -> None:
    print(
        "Breakout controls: Space to serve, A/D or Left/Right to move, "
        "R to restart, Esc/Q to quit."
    )
    env = BreakoutEnv(render_mode=True)
    env.reset()

    try:
        running = True
        while running:
            running, fire, reset = _process_events()
            if not running:
                break
            if reset:
                env.reset()
                continue

            action = 1 if fire else _breakout_action()
            _, _, done, info = env.step(action)
            if done:
                print(f"Game Over! Score: {info['score']}")
                env.reset()
    finally:
        env.close()


def play_flappy() -> None:
    print("Flappy Bird controls: Space to flap, R to restart, Esc/Q to quit.")
    env = FlappyBirdEnv(render_mode=True)
    env.reset()

    try:
        running = True
        while running:
            running, flap, reset = _process_events()
            if not running:
                break
            if reset:
                env.reset()
                continue

            _, _, done, info = env.step(1 if flap else 0)
            if done:
                print(f"Game Over! Score: {info['score']}")
                env.reset()
    finally:
        env.close()


def main() -> None:
    args = parse_args()
    if args.game == "breakout":
        play_breakout()
    else:
        play_flappy()


if __name__ == "__main__":
    main()
