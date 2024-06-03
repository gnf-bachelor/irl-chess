from selenium import webdriver
from selenium.webdriver.common.by import By
from tqdm import tqdm
import numpy as np
import time
import json
from bs4 import BeautifulSoup
import chess.pgn
from irl_chess.misc_utils.load_save_utils import get_board_after_n
from irl_chess.models.sunfish_GRW import sunfish_move
from irl_chess.chess_utils.sunfish_utils import get_new_pst, str_to_sunfish_move, sunfish_move_to_str
from irl_chess.chess_utils.sunfish_utils import board2sunfish, sunfish2board
from maia_chess import load_maia_network

def wait_until(driver, string, time=5):
    waited = 0
    while string not in str(driver.page_source):
        time.sleep(0.2)
        waited += 1
        if waited > 25:
            break


def configure(driver, set_time_controls, set_ratings, first=False):
    all_ratings = ['400', '1000', '1200', '1400', '1600', '1800', '2000', '2200', '2500']
    all_time_controls = ['UltraBullet', 'Bullet', 'Blitz', 'Rapid', 'Classical', 'Correspondence']
    if first:
        driver.find_element(By.XPATH, '//*[@title="Opening explorer & tablebase"]').click()
        time.sleep(0.5)
        driver.find_element(By.XPATH, '//*[@title="Rated games sampled from all Lichess players"]').click()
        time.sleep(0.5)
    time.sleep(0.5)
    success = False
    while not success:
        try:
            driver.find_element(By.XPATH, '//*[@aria-label="Open configuration"]').click()
            success = True
        except:
            continue
    time.sleep(0.5)

    for time_control in all_time_controls:
        button = driver.find_element(By.XPATH, f'//*[@title="{time_control}"]')
        button_on = button.get_attribute("aria-pressed") == 'true'
        rating_included = time_control in set_time_controls
        if not button_on == rating_included:
            button.click()
            time.sleep(0.5)

    order = 1 if set_ratings[-1] == '1200' else -1
    for rating in all_ratings[::order]:
        button = driver.find_element(By.XPATH, f"//*[text()={rating}]")
        button_on = button.get_attribute("aria-pressed") == 'true'
        rating_included = rating in set_ratings
        if not button_on == rating_included:
            button.click()
            time.sleep(0.5)

    driver.find_element(By.XPATH, '//*[@aria-label="Close configuration"]').click()

def get_player_move(driver, ratings):
    configure(driver, time_controls, ratings)
    waited = 0
    while 'database' not in str(driver.page_source):
        time.sleep(0.2)
        waited += 1
        if waited > 25:
            driver.close()
            driver = webdriver.Chrome()
            driver.implicitly_wait(1)
            driver.get(url)
            configure(driver, time_controls, ratings, first=True)

    time.sleep(0.5)
    soup = BeautifulSoup(driver.page_source)
    tables = soup.find_all("table", {"class": "moves"})

    try:
        rows = tables[0].find_all("tr")[1:]
    except IndexError:
        return None

    move = rows[0]['data-uci']
    return move

R_low = np.array([100,190,190,210,300,65000])
pst_low = get_new_pst(R_low)

R_high = np.array([100,160,200,250,320,65000])
pst_high = get_new_pst(R_high)

maia_low = load_maia_network(1100, parent='maia_chess/')
maia_high = load_maia_network(1900, parent='maia_chess/')

time_controls = ['Blitz', 'Rapid', 'Classical']
ratings_low = ['1000','1200']
ratings_high = ['1800','2000']
n_boards = 10000

site = 'https://lichess.org/analysis/standard/'
driver = webdriver.Chrome()
driver.implicitly_wait(1)

pgn = open("data/raw/lichess_db_standard_rated_2013-01.pgn")
move_dicts = {}

pbar = tqdm(total=n_boards, desc='Getting position info')

driver.get(site)
configure(driver, time_controls, ratings_low, first=True)

start_from = 1200
for i in range(start_from):
    game = chess.pgn.read_game(pgn)

pbar.update(start_from)

while len(move_dicts) < n_boards:
    pbar.update(1)
    game = chess.pgn.read_game(pgn)
    board = game.board()
    state, player_move = get_board_after_n(game, 14)

    # model moves
    sf_state = board2sunfish(state, 0)
    sunfish_move_low = sunfish_move(sf_state, pst_low, time_limit=0.5, move_only=True)
    sunfish_move_high = sunfish_move(sf_state, pst_high, time_limit=0.5, move_only=True)
    sunfish_move_low, sunfish_move_high = sunfish_move_to_str(sunfish_move_low), sunfish_move_to_str(sunfish_move_high)

    maia_move_low = maia_low.getTopMovesCP(state, 1)[0][0]
    maia_move_high = maia_high.getTopMovesCP(state, 1)[0][0]

    fen = state.fen()
    url = site + fen
    driver.get(url)

    move_low = get_player_move(driver, ratings_low)
    move_high = get_player_move(driver, ratings_high)

    if move_low == None or move_high == None:
        continue

    player_difference = move_low != move_high
    sunfish_difference = sunfish_move_low != sunfish_move_high
    maia_difference = maia_move_low != maia_move_high

    if player_difference:
        print(f'Game: {fen}')
        print(f'Player: low={move_low}, high={move_high}')
        print(f'Maia: low={maia_move_low}, high={maia_move_high}')
        print(f'Sunfish: low={sunfish_move_low}, high={sunfish_move_high}')
    else:
        print(f'Player: low={move_low}, high={move_high}')

