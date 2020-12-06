
import cProfile
import numpy as np
import cv2
from mss import mss
from PIL import Image, ImageGrab
import glob
from AppKit import NSApplication, NSApp, NSWorkspace
from Quartz import kCGWindowListOptionOnScreenOnly, kCGNullWindowID, CGWindowListCopyWindowInfo
import osascript
import pytesseract
import time
from datetime import datetime
import logging
import tensorflow as tf

# Remove tensorflow info.
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

from algo import Algo, Card

logging.basicConfig(filename = "logging/" + datetime.now().strftime("%Y-%m-%d"),
        level = logging.DEBUG, format = '%(asctime)s:%(levelname)s:%(message)s', datefmt = 
        '%H:%M:%S')


colors = ["H", "C", "S", "D"]
all_cards = sorted(np.array([str(value) + color for color in colors for value
    in range(2, 15)]))


def show(img):
    """
    Display image.
    """
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    cv2.imshow("card", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


import contextlib
@contextlib.contextmanager
def timeit(title = ""):
    tstart = time.time()
    yield
    elapsed = time.time() - tstart
    print(title, elapsed)


class Board():
    """
    Class over a single board.
    """
    def __init__(self, name = None, pre_board = None):
        super().__init__()
        
        self.name = name
        logging.info("\n")
        logging.info("Joining new board: {}".format(self.name))
        self.pid = None
        self.bounds = dict()
        self.image = None
        self.image_grey = None
        self.open_cards = []
        self.pott = 0
        self.card_model = tf.keras.models.load_model('model/PokerModel_bigtable')

        # If using predefined board, easier for testing, doesn't need PS live.
        if pre_board:
            image = Image.open(pre_board)
            self.image = np.asarray(image)
            self.image_grey = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)

        else:
            self.set_position()
            self.position_info()
            self.get_board_image()
    

        # Initialize players
        self.player = MainPlayer(self)

        # 6 player deck
        self.opponents = []
        position_opponents = {
                1: [(520, 650), (38, 238)],
                2: [(195, 325), (85, 285)],
                3: [(95, 225), (605, 805)],
                4: [(195, 325), (1215, 1415)],
                5: [(520, 650), (1262, 1462)]
                }
        
        for number, pos in position_opponents.items():
            self.opponents.append(Opponent(self, number, pos))
        

    def get_board_image(self):
        """
        Fetch image of the board.
        """
        with mss() as sct:
            def to_rgb(im):
                frame = np.array(im, dtype=np.uint8)
                return np.flip(frame[:, :, :3], 2)

            monitor = {"top": self.bounds["Y"], "left": self.bounds["X"], 
                    "width": self.bounds["Width"], "height": self.bounds["Height"]}
            im = sct.grab(monitor)
            self.image = to_rgb(im)
            self.image_grey = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)


        #im = Image.fromarray(self.image)
        #im.save("board_large.png")

    
    def update_stats(self):
        """
        Updates the board information, i.e., pott and open cards.
        """
        self._clear_board()
        self._update_pott()
        self._update_open_cards()

        # Update main player
        self.player.update_stats()

        # Update opponents stats
        for opponent in self.opponents:
            opponent.update_stats()
    
    def _clear_board(self):
        """
        Helper method to clear the board before updating, i.e.,
        when a update is needed, the previous values are clear as a precaution.
        """
        self.open_cards = []


    def number_open_cards(self):
        """ 
        Get number of open cards on the table. This only
        accounts for 1-5 open cards, not when multiple more are shown.
        """
        num_open = 0
        offset = 100
        for ii in range(5):
            col_start = 569 + ii*offset
            col_end = col_start + 1
            pixel = self.image[399:400, col_start:col_end, :]
            if (pixel[0, 0, 0] > 50 and pixel[0, 0, 1] > 50 and 
                    pixel[0, 0, 2] > 50):
                num_open += 1

        number_open_cards = num_open
        logging.info("Number of open cards: {}".format(number_open_cards))
        return number_open_cards

        
    def _update_open_cards(self):
        """
        Updates the open cards on the table.
        """
        number_open_cards = self.number_open_cards()
        c1 = self.image[390:440, 525:565][np.newaxis, ...]
        c2 = self.image[390:440, 618:658][np.newaxis, ...]
        c3 = self.image[390:440, 711:751][np.newaxis, ...]
        c4 = self.image[390:440, 804:844][np.newaxis, ...]
        c5 = self.image[390:440, 897:937][np.newaxis, ...]
        cards = [c1, c2, c3, c4, c5]

        for ii in range(number_open_cards):
            predictions = self.card_model.predict(cards[ii])
            score = tf.nn.softmax(predictions[0])
            card = all_cards[np.argmax(score)]
            confidence = round(100 * np.max(score), 2)
            self.open_cards.append(Card(card))

        logging.info("Open cards on table: {}".format(self.open_cards))


    def _update_pott(self):
        """
        Retrieves the pot on the table.
        """
        # This dimensions are for the small board.
        #pott_img_grey_smallboard = self.image_grey.copy()[210:254, 380:570]
        pott_img_grey = self.image_grey.copy()[340:380, 650:850]
        temp = pott_img_grey.copy()
        pott_img_grey[pott_img_grey <= 130] = 255
        pott_img_grey[temp >= 130] = 0
        pott_grey = pytesseract.image_to_string(pott_img_grey).strip()
        logging.info("Pott tesseract: {}".format(pott_grey))
        try:
            self.pott = pott_grey.split("$")[1].replace(" ", "").replace(",", ".")

        except:
            logging.exception("")


    def _get_pid(self):
        options = kCGWindowListOptionOnScreenOnly
        windows = CGWindowListCopyWindowInfo(options,
            kCGNullWindowID)

        for window in windows:
            window_name = window["kCGWindowName"]
            # Can't use two of the same tables, e.g., Aaryn and Aaryn II
            if self.name in window_name:
                self.name = window_name
                self.pid = int(window["kCGWindowOwnerPID"])


    def set_position(self):
        self._get_pid()
        if self.pid:
            #set pid to "15161"
            #set proc to item 1 of (processes whose unix id is pid)
                #tell proc
                    #set position of front window to {1, 1}
                #end tell
            #end tell

            #tell application "PokerStarsSE" to get the bounds of the window 1
            code, out, err = osascript.run('''
                tell application "System Events" to tell application process "PokerStarsSE"
                    tell windows 
                        set {size} to {{750, 950}}
                    end tell
                end tell
                ''')
            #print(code, out, err)
    
    def position_info(self):
        """
        Get position and bounds of the whole board window.
        """
        options = kCGWindowListOptionOnScreenOnly
        windows = CGWindowListCopyWindowInfo(options,
            kCGNullWindowID)

        for window in windows:
            window_name = window["kCGWindowName"]
            # Can't use two of the same tables, e.g., Aaryn and Aaryn II
            if self.name in window_name:
                self.name = window_name
                bounds = window["kCGWindowBounds"]
                for key, value in bounds.items():
                    self.bounds[key] = int(value)




class MainPlayer():
    def __init__(self, board):
        self.board = board
        self.balance = 0
        self.name = ""
        self.number = 1
        self.cards = np.array([])

        self._update_name()

    def update_stats(self):
        """
        Helper function to call all other update methods.
        """
        self._update_balance()
        self._update_hand()


    def action_needed(self):
        """
        Checks if it is the main players turn by checking a single pixel
        in the yellow progress bar below.
        Returns:
            True/False
        """
        pixel = self.board.image[854:855, 715:716]
        if not (pixel[0, 0, 0] < 50 and pixel[0, 0, 1] and 
                pixel[0, 0, 2] < 50):
            return True

        return False
       

    def _update_hand(self):
        """
        Get the open hand for the main player.
        Image needs to be (85x60)
        """
        # Clear cards before updating.
        self.cards = []

        h1 = self.board.image[700:750, 668:708]
        h2 = self.board.image[700:750, 756:796]
        h1 = h1[np.newaxis, ...]
        h2 = h2[np.newaxis, ...]
        hs = [h1, h2]
        for h in hs:
            predictions = self.board.card_model.predict(h)
            score = tf.nn.softmax(predictions[0])
            card = all_cards[np.argmax(score)]
            confidence = round(100 * np.max(score), 2)
            self.cards.append(Card(card))
        
        logging.info("Main player cards: {}".format(self.cards))

        # Big table save images
        """
        im1 = Image.fromarray(h1)
        im2 = Image.fromarray(h2)
        r1 = np.random.randint(1000000)
        r2 = np.random.randint(1000000)
        im1.save("model/data_bigtable_unsorted/" + str(r1) + ".jpeg")
        im2.save("model/data_bigtable_unsorted/" + str(r2) + ".jpeg")
        """

    def _update_name(self):
        """
        Uses tesseract to get the mainplayers name.
        """
        img = self.board.image_grey[760:792, 710:890]
        img = cv2.bitwise_not(img)
        ret, thresh = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        morphed = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, 
            cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1)))
        up_scaled = np.ones((morphed.shape[0]*3, morphed.shape[1]*3), dtype = 
                'uint8')*255
        up_scaled[morphed.shape[0]:morphed.shape[0]*2, morphed.shape[1]:
                morphed.shape[1]*2] = morphed
        name = pytesseract.image_to_string(up_scaled).strip()
        self.name = name
        logging.info("Main player name: {}".format(self.name))

    def _update_balance(self):
        """
        Updates the main player's current balance.
        """
        # Img for smallest board
        #img = self.board.image_grey[523:546, 444:560]
        img = self.board.image_grey[800:835, 710:870]
        img = cv2.bitwise_not(img)

        # Thresholding, using OTSU since the alpha of the images are changing.
        ret, thresh = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        morphed = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, 
            cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1)))

        up_scaled = np.ones((morphed.shape[0]*3, morphed.shape[1]*3), 
                dtype = 'uint8')*255
        up_scaled[morphed.shape[0]:morphed.shape[0]*2, morphed.shape[1]:
                morphed.shape[1]*2] = morphed
        img_balance = pytesseract.image_to_string(up_scaled).strip().lower()
        logging.info("Main player balance: {}".format(img_balance))

        if img_balance in ["all-in", "allhn", "aihn", "cash in", "cash out"]:
            self.allin = True

        else: 
            try:
                self.balance = img_balance.split("$")[1].replace(" ", 
                    "").replace(",", ".")
            except:
                logging.exception("")



class Opponent():
    """
    Representing all of the opponents.
    """
    def __init__(self, board, number, position):
        self.board = board
        # Opponent number, goes clockwise starting from left of player.
        self.number = number
        self.position = position
        self.balance = 0
        self.name = ""
        self.allin = False
        self.image = None
        self.folded = False

    def update_stats(self):
        """
        Helper function to call all other update methods.
        Begins by updating the opponents image, then all other methods
        uses the subimage from the whole board to retrieve information.
        """
        self._update_img()
        self._update_name()
        self._update_balance()
        self._have_folded()

    def restore_information(self):
        """
        Restore an opponents attributes to base values.
        """
        self.name = ""
        self.balance = 0
        self.allin = False
        self.image = None
        self.folded = False


    def _update_img(self):
        """
        Updates the opponents part of the board.
        """
        self.image = self.board.image[self.position[0][0]:self.position[0][1],
                self.position[1][0]:self.position[1][1]]
        self.image_grey = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)

    def _update_balance(self):
        """
        Updates the opponent's current balance.
        """
        img = cv2.bitwise_not(self.image_grey)
        img_balance = img[90:125, 10:190]
        
        ret, thresh = cv2.threshold(img_balance, 0, 255, 
                cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        morphed = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, 
            cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1)))
        up_scaled = np.ones((morphed.shape[0] * 3, morphed.shape[1] * 3), 
                dtype = 'uint8') * 255
        up_scaled[morphed.shape[0]:morphed.shape[0] * 2, morphed.shape[1]:
                morphed.shape[1] * 2] = morphed

        balance_string = pytesseract.image_to_string(up_scaled, 
                config = '--psm 6').strip().lower()
        logging.info("Opponent {} balance string tesseract: {}".format(
            self.number, balance_string))

        if balance_string in ["all-in", "allhn", "aihn", "cash in", "cash out"]:
            self.allin = True
        
        else:
            try:
                balance = balance_string.split("$")[1]
                self.balance = balance.replace(" ", "").replace(",", ".")
                logging.info("Opponent {} balance: {}".format(self.number, 
                    self.balance))

            except:
                logging.exception("Opponent {}".format(self.number))
        

    def _update_name(self):
        """
        Adds the opponents name.
        """
        img = cv2.bitwise_not(self.image_grey)
        img_name = img[50:86, 5:195]
        ret, thresh = cv2.threshold(img_name, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        morphed = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, 
            cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1)))
        up_scaled = np.ones((morphed.shape[0]*3, morphed.shape[1]*3), 
                dtype = 'uint8')*255
        up_scaled[morphed.shape[0]:morphed.shape[0]*2, morphed.shape[1]:
                morphed.shape[1]*2] = morphed
        
        name_string = pytesseract.image_to_string(up_scaled, config = 
                '--psm 6').strip().lower()
        logging.info("Opponent {} name string tesseract: {}".format(self.number, name_string))

        if name_string not in ["raise", "fold", "post bb", "post sb", "resume", 
                "call", "check", "bet", "cash out"]:
            self.name = name_string
            logging.info("Opponent {} name: {}".format(self.number, self.name))

    
    def _have_folded(self):
        """
        Check a single pixel, if white, opponent haven't folded.
        """
        pixel = self.image[30:31, 110:111, :]
        if (pixel[0, 0, 0] < 40 and pixel[0, 0, 1] < 40 and 
                pixel[0, 0, 2] < 40):
            self.folded = True

        logging.info("Opponent {} have folded: {}".format(self.number, self.folded))



def start_game(board):
    algo = Algo()
    updated = False

    while (not time.sleep(0.5)):
        board.get_board_image()
        if (board.player.action_needed()):
            if not updated:
                board.update_stats()
                algo.analyze(board)
                updated = True

        else:
            updated = False



def main():
    #board = Board("Orthos")
    #start_game(board)

    with timeit():
        board = Board(pre_board = "boards/board_large.png")
        board.update_stats()
        algo = Algo()
        algo.analyze(board)




if __name__ == '__main__':
    main()


