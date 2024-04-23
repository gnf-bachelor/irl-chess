from selenium import webdriver
from selenium.webdriver.common.by import By
from tqdm import tqdm
import time
import json
from bs4 import BeautifulSoup
import chess.pgn
from irl_chess.misc_utils.load_save_utils import get_board_after_n

def configure(driver, set_time_controls, set_ratings):
    all_ratings = ['400', '1000', '1200', '1400', '1600', '1800', '2000', '2200', '2500']
    all_time_controls = ['UltraBullet', 'Bullet', 'Blitz', 'Rapid', 'Classical', 'Correspondence']
    driver.find_element(By.XPATH, '//*[@title="Opening explorer & tablebase"]').click()
    time.sleep(0.5)
    driver.find_element(By.XPATH, '//*[@title="Rated games sampled from all Lichess players"]').click()
    time.sleep(0.5)
    driver.find_element(By.XPATH, '//*[@aria-label="Open configuration"]').click()
    time.sleep(0.5)

    for time_control in all_time_controls:
        button = driver.find_element(By.XPATH, f'//*[@title="{time_control}"]')
        button_on = button.get_attribute("aria-pressed") == 'true'
        rating_included = time_control in set_time_controls
        if not button_on == rating_included:
            button.click()
            time.sleep(0.5)

    for rating in all_ratings:
        button = driver.find_element(By.XPATH, f"//*[text()={rating}]")
        button_on = button.get_attribute("aria-pressed") == 'true'
        rating_included = rating in set_ratings
        if not button_on == rating_included:
            button.click()
            time.sleep(0.5)

    driver.find_element(By.XPATH, '//*[@aria-label="Close configuration"]').click()


# Categories to be excluded
time_controls = ['Blitz', 'Rapid', 'Classical']
ratings = ['1000','1200']
n_boards = 10000

site = 'https://lichess.org/analysis/standard/'
driver = webdriver.Chrome()
driver.implicitly_wait(1)

pgn = open("data/raw/lichess_db_standard_rated_2013-01.pgn")
move_dicts = {}

save_point = 5269
file_path = f'data/move_percentages/moves_sub1200_{save_point}'
with open(file_path, "r") as f:
    loaded_dict = json.load(f)

loaded_boards = list(loaded_dict.keys())
pbar = tqdm(total=save_point, desc='Getting to save point')
while pbar.n < save_point:
    game = chess.pgn.read_game(pgn)
    board = game.board()
    state, player_move = get_board_after_n(game, 12)
    fen = state.fen()
    if fen in loaded_boards:
        pbar.update(1)
print(f'Loaded to save point:{pbar.n}/{save_point}')

pbar = tqdm(total=n_boards, desc='Getting position info')
pbar.update(save_point)
driver.get(site)
configure(driver, time_controls, ratings)

while len(move_dicts) < n_boards:
    game = chess.pgn.read_game(pgn)
    board = game.board()
    state, player_move = get_board_after_n(game, 12)

    fen = state.fen()
    url = site + fen
    driver.get(url)

    waited = 0
    while 'database' not in str(driver.page_source):
        time.sleep(0.2)
        waited += 1
        if waited > 25:
            driver.close()
            driver = webdriver.Chrome()
            driver.implicitly_wait(1)
            driver.get(url)
            configure(driver, time_controls, ratings)

    soup = BeautifulSoup(driver.page_source)
    tables = soup.find_all("table", {"class": "moves"})

    try:
        rows = tables[0].find_all("tr")[1:]
        if len(rows) < 2:
            continue
    except IndexError:
        continue

    moves = {}
    for row in rows:
        move = row['data-uci']
        if move == '':
            move = 'sum'
        info = row.find_all("td")
        fraction = int(info[1].text[:-1])/100
        count = int(info[2].text.replace(',', ''))
        moves[move] = [fraction, count]
    if moves['sum'][1] < 20:
        continue

    move_dicts[fen] = moves
    pbar.update(1)

file_path = f'data/move_percentages/moves_sub1200_{pbar.n}'
with open(file_path, 'w') as f:
    json.dump(move_dicts, f)

with open(file_path, "r") as f:
    loaded_data = json.load(f)
