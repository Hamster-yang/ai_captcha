from selenium import webdriver
from selenium.webdriver.common.by import By
import pyautogui

driver = webdriver.Edge()
driver.get('http://127.0.0.1/captcha_index.php')

#driver.save_screenshot('test.png')
while(1):
    element = driver.find_element(by=By.NAME,value='checkword')
    
    print(element.location)
    print(element.size)
    pyautogui.moveTo(element.location['x']+5, element.location['y'], duration = 10)
    pyautogui.click(clicks=1, interval=0.5, button='left')
    pyautogui.press('t') #按下c鍵
    pyautogui.press('backspace') #按下c鍵