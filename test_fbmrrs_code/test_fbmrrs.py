import sys, os
import unittest 
from utils.fbmrrs import DataCollection
from unittest.mock import patch, Mock, call
from time import sleep

class DataCollectionTest(unittest.TestCase):

    def setUp(self) -> None:
        self.scraper_obj = DataCollection(url="")
        self.scraper_obj.pass_cookies()
        return super().setUp()



    def tearDown(self) -> None:
        self.scraper_obj.driver.quit()
        return super().tearDown()

if __name__ == "__main__":
    unittest.main(argv=[""], verbosity=3, exit=True)