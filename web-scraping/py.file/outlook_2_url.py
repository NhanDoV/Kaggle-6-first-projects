# ### 0. Packages Installation!
# !pip install selenium
# !pip install parsel
# !apt-get update # to update ubuntu to correctly run apt install
# !apt install -y chromium-chromedriver
#----------------------------------------------------------------------------------------#
## 1. Import libraries & sign in hotmail account
##################### 1.1 import your library ======================
import time
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
t0 = time.time()
driver = r'C:\Users\Admin\Downloads\chromedriver_win32\chromedriver' # should be replaced by the path to your chromedriver
driver = webdriver.Chrome(driver)
######################################## 1.2. Log in to hotmail ===========================
driver.get('https://outlook.live.com/owa/?nlp=1')
"""----------------------------------------------------------------------------------"""
# sign in to outlook; first run
username = driver.find_element_by_name('loginfmt')
username.send_keys('thanosteamk09@gmail.com')
log_in_button = driver.find_element_by_id('idSIButton9')
log_in_button.click()
"""----------------------------------------------------------------------------------"""
## second run; enter your password
password = driver.find_element_by_name('passwd')
password.send_keys('thaNos99*')
log_in_button = driver.find_element_by_id('idSIButton9')
log_in_button.click()
"""----------------------------------------------------------------------------------"""
## 3rd step (run if sign in the first time). Click to "Don't show this again" and choose the button "Yes"
log_in_button = driver.find_element_by_id('KmsiCheckboxField')
log_in_button.click()
## 4th step
log_in_button = driver.find_element_by_id('idSIButton9')
log_in_button.click()
## 5th line
log_in_button = driver.find_element_by_id('KmsiCheckboxField')
log_in_button.click()
## 6th line
log_in_button = driver.find_element_by_id('idSIButton9')
log_in_button.click()
"""----------------------------------------------------------------------------------"""
## Now, enter to email_contact_address
driver.get('https://outlook.live.com/people/0/')

#----------------------------------------------------------------------------------------#
# 2. loop through email list to extract url_linkedin_link
## Example 1. look at the first email address then find the linkedin-url
### Step 1. find element by xpath from the email address
html_full_xpath = "/html/body/div[2]/div/div[2]/div/div[3]/div/div[2]/div/div[1]/div/div[2]/div[1]/div/div/div/div/div[2]/div/div/div[2]"
contact_button = driver.find_element_by_xpath(html_full_xpath)
contact_button.click()
### Step 2. Click to linkedin-tab
linkedin_xpath = "/html/body/div[2]/div/div[2]/div/div[3]/div/div[2]/div/div[3]/div/div/div/div/div[1]/div/div/div/button[4]/span/span/span"
contact_button = driver.find_element_by_xpath(linkedin_xpath)
contact_button.click()
### Step 3. Move to linkedin url from the hyperlink "See full profile on Linkedin"
url_linkedin_link_xpath = "/html/body/div[2]/div/div[2]/div/div[3]/div/div[2]/div/div[3]/div/div/div/div/div[2]/section/div[2]/button"
contact_button = driver.find_element_by_xpath(url_linkedin_link_xpath)
contact_button.click()
## Step 4.
driver.get('https://www.linkedin.com')
username = driver.find_element_by_name('session_key')
username.send_keys('thanosteamk09@gmail.com')
password = driver.find_element_by_name('session_password')
password.send_keys('thaNos99*')
log_in_button = driver.find_element_by_class_name('sign-in-form__submit-button')
log_in_button.click()


### 2,2, Now. We find the different and rules of each html_full_xpath
first_html = "/html/body/div[2]/div/div[2]/div/div[3]/div/div[2]/div/div[1]/div/div[2]/div[1]/div/div/div/div/div[2]/div/div/div[2]/div[2]/div/div/div"
second_html = "/html/body/div[2]/div/div[2]/div/div[3]/div/div[2]/div/div[1]/div/div[2]/div[1]/div/div/div/div/div[3]/div/div/div[2]/div[2]/div/div/div"

## Rule: "/html/body/div[2]/div/div[2]/div/div[3]/div/div[2]/div/div[1]/div/div[2]/div[1]/div/div/div/div/" + "div[line]" + "/div/div/div[2]/div[2]/div/div/div"