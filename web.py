from selenium import webdriver
from selenium.webdriver.common.by import By
import pyautogui

driver = webdriver.Edge()
driver.get('127.0.0.1/captcha_index.php')

#driver.save_screenshot('test.png')
while(1):

    element = driver.find_element(by=By.NAME,value='checkword')
    element.send_keys("Test")
