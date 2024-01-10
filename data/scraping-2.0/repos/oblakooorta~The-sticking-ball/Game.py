import pygame

from Ball import Ball
from Guidance import Guidance
from Goal import Goal
from Menu import Menu
from Level_1 import platforms_1
from Level_1 import elastic_1
from Level_1 import disappearing_1
from Level_1 import moving_v_1
from Level_1 import moving_h_1
from Level_1 import death_1
from Level_1 import x_started1, y_started1, x_end1, y_end1

from Level_2 import platforms_2
from Level_2 import elastic_2
from Level_2 import disappearing_2
from Level_2 import moving_v_2
from Level_2 import moving_h_2
from Level_2 import death_2
from Level_2 import x_started2, y_started2, x_end2, y_end2

FPS = 30
RED = 0xFF0000
BLUE = 0x0000FF
YELLOW = 0xFFC91F
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GREEN = 0x00FF00
MAGENTA = 0xFF03B8
CYAN = 0x00FFCC
GREY = 0x7D7D7D

WIDTH = 1280
HEIGHT = 720


pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
clock = pygame.time.Clock()

points = 0
finished = False
game_started = False
game_paused = False
next = False
win_L1 = False
win_L2 = False

ball_1 = Ball(screen, x_started1, y_started1)
ball_2 = Ball(screen, x_started2, y_started2)

finish_1 = Goal(screen, x_end1, y_end1)
finish_2 = Goal(screen, x_end2, y_end2)

arrow_1 = Guidance(screen, ball_1)
arrow_2 = Guidance(screen, ball_2)

start_button = Menu(540, 310, 200, 100, RED, "Начать")
pause_button = Menu(1150, 30, 80, 40, RED, "Пауза")
win_text = Menu(540, 260, 200, 100, RED, "Победа!")
next_level = Menu(520, 400, 240, 100, RED, "Следующий уровень")
exit_button = Menu(540, 400, 200, 100, RED, "Выход")


while not win_L2:
    screen.fill(WHITE)
    if not game_started and not win_L1:
        start_button.draw(screen)
    elif game_started and (not win_L1 or not win_L2):
        pause_button.draw(screen)
    elif win_L1:
        start_button.drawn = False

    if not next and win_L1:
        win_text.draw(screen)
        next_level.draw(screen)
        game_started = False

    if game_started:
        arrow_1.update(ball_1)
        arrow_1.draw()
        ball_1.draw()
        finish_1.draw()
        ball_1.move()

        for mov in moving_v_1:
            mov.draw()
            mov.move_vertically()
            mov.precollision(ball_1)
            if mov.collision(ball_1):
                ball_1.sticking()
                ball_1.move_v(mov)

        for mov in moving_h_1:
            mov.draw()
            mov.move_horizontally()
            mov.precollision(ball_1)
            if mov.collision(ball_1):
                ball_1.sticking()
                ball_1.move_h(mov)

        for pl in platforms_1:
            pl.draw()
            pl.precollision(ball_1)
            if pl.collision(ball_1):
                ball_1.sticking()

        for el in elastic_1:
            el.draw()
            el.precollision(ball_1)
            if el.collision(ball_1):
                ball_1.jumping_back(el)

        for dis in disappearing_1:
            dis.draw()
            dis.precollision(ball_1)
            if dis.collision(ball_1):
                disappearing_1.remove(dis)
                ball_1.fall()

        for d in death_1:
            d.draw()
            if d.collision(ball_1):
                ball_1.x = x_started1
                ball_1.y = y_started1

        if finish_1.collision(ball_1):
            win_L1 = True

    if game_paused:
        game_started = not game_started
        game_paused = not game_paused

    if win_L2:
        win_text.draw(screen)
        exit_button.draw(screen)

    if next:
        game_started = False
        arrow_2.update(ball_2)
        arrow_2.draw()
        finish_2.draw()
        ball_2.draw()
        ball_2.move()

        for mov in moving_v_2:
            mov.draw()
            mov.move_vertically()
            mov.precollision(ball_2)
            if mov.collision(ball_2):
                ball_2.sticking()
                ball_2.move_v(mov)

        for mov in moving_h_2:
            mov.draw()
            mov.move_horizontally()
            mov.precollision(ball_2)
            if mov.collision(ball_2):
                ball_2.sticking()
                ball_2.move_h(mov)

        for pl in platforms_2:
            pl.draw()
            pl.precollision(ball_2)
            if pl.collision(ball_2):
                ball_2.sticking()

        for el in elastic_2:
            el.draw()
            el.precollision(ball_2)
            if el.collision(ball_2):
                ball_2.jumping_back(el)

        for dis in disappearing_2:
            dis.draw()
            dis.precollision(ball_2)
            if dis.collision(ball_2):
                disappearing_2.remove(dis)
                ball_2.fall()

        for d in death_2:
            d.draw()
            if d.collision(ball_2):
                ball_2.x = x_started2
                ball_2.y = y_started2

        if finish_2.collision(ball_2):
            win_L2 = True
            finished = True

    pygame.display.update()
    pygame.display.flip()

    clock.tick(FPS)

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
        elif event.type == pygame.MOUSEMOTION:
            if not win_L1:
                arrow_1.targetting(event)
            if win_L1:
                arrow_2.targetting(event)
        elif event.type == pygame.MOUSEBUTTONDOWN:
            if start_button.rect.collidepoint(event.pos):
                if start_button.drawn:
                    game_started = True
            elif pause_button.rect.collidepoint(event.pos):
                if pause_button.drawn:
                    game_paused = True
            elif next_level.rect.collidepoint(event.pos):
                if next_level.drawn:
                    next = True
            elif not win_L1:
                arrow_1.fire2_start(ball_1)
            elif win_L1:
                arrow_2.fire2_start(ball_2)
        elif event.type == pygame.MOUSEBUTTONUP:
            if not win_L1:
                ball_1.jump(event, arrow_1)
                arrow_1.fire2_end(ball_1)
            elif win_L1:
                ball_2.jump(event, arrow_2)
                arrow_2.fire2_end(ball_2)
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_r:
                if not win_L1:
                    ball_1.x = x_started1
                    ball_1.y = y_started1
                elif win_L1:
                    ball_2.x = x_started2
                    ball_2.y = y_started2
    if not win_L1:
        arrow_1.power_up()
    elif win_L1:
        arrow_2.power_up()

esc = False

if win_L2:
    screen.fill(WHITE)
    exit_button.draw(screen)
    win_text.draw(screen)
    pygame.display.update()
    while True:
        for event in pygame.event.get():
            if event.type == pygame.MOUSEBUTTONDOWN:
                if exit_button.rect.collidepoint(event.pos):
                    pygame.quit()

else:
    screen.fill(BLACK)
    pygame.display.update()
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
