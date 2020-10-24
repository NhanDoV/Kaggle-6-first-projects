# ### 0. Packages Installation!
# !pip install selenium
# !pip install parsel
# **Install chrome**
# !apt-get update # to update ubuntu to correctly run apt install
# !apt install -y chromium-chromedriver

# ### 1.1 Opening Linkedin with `webdriver.Chrome()`
import time 
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
t0 = time.time()
driver = r'C:\Users\Admin\Downloads\chromedriver_win32\chromedriver' # should be replaced by the path to your chromedriver
driver = webdriver.Chrome(driver)
driver.get('https://www.linkedin.com')
username = driver.find_element_by_name('session_key')
username.send_keys('thanosteamk09@gmail.com')
password = driver.find_element_by_name('session_password')
password.send_keys('thaNos99*')
log_in_button = driver.find_element_by_class_name('sign-in-form__submit-button')
log_in_button.click()

# ### 1.2. Import the `Selector` from `parsel` for scrapping linkedin accounts.
## Step 1. Idea of stoping-time-in-while,
for k in range(20):
    t0 = time.time()                         ## index in your iteration
    fk = np.uint8(k**3 + k**2 + k + 1)       ## function defined w.r.t the index
    fk = fk // 121 
    print('k = %s, fk = %s, time_k = %s'%(k, fk, time.time() - t0));  ## measure the time-out of function in each step
    if (k+1)%10 == 0:
        t0 = time.time()
        wait_time = 5   ## timeout variable [seconds] can be omitted, if you use specific value in the while condition
        while time.time() < t0 + wait_time:
            stop = 0
            if stop == 5:
                break
            stop -= 1
        print('stoping_time = ',time.time() - t0)
    elif k==20:
        print('Done! complete the short-testing!')

##### Step 2. Which `url link` is used to scrap?
from parsel import Selector
## List of url link
links = ['https://www.linkedin.com/in/do-nhan-39a4191a3/',
        'https://uk.linkedin.com/in/thornbeck',
        'https://www.linkedin.com/in/brandon-foxworth-793a4980/',
        'https://www.linkedin.com/in/steve-benckenstein/',
        'https://www.linkedin.com/in/minh-hieu-do-82175740/',
         'https://uk.linkedin.com/in/pauljgarner', 
         'https://www.linkedin.com/in/lacy-judd-ba845273/',
         'https://www.linkedin.com/in/daniel-reyes-367b64b2/',
         'https://www.linkedin.com/in/thoa-thieu-294252135/',
         'https://uk.linkedin.com/in/eastwoodalex', 
         'https://www.linkedin.com/in/lewis-forsyth-49333284/',
         'https://www.linkedin.com/in/hoang-ha-84b31041/',
         'https://uk.linkedin.com/in/navaneetham', 
         'https://www.linkedin.com/in/ltp238/',
         'https://www.linkedin.com/in/hoan-huynh-8b3ba9b6', 
         'https://www.linkedin.com/in/yoelohayon', 
         'https://www.linkedin.com/in/tknguyen2015/',
         'https://www.linkedin.com/in/siddharthsatpathy',
         'https://www.linkedin.com/in/thien-trang-bui/',
         'https://www.linkedin.com/in/david-williams-9a1392153/',
         'https://www.linkedin.com/in/jonathanhvan/',
         'https://www.linkedin.com/in/stephen-long-b786b71ab/',
         'https://www.linkedin.com/in/kayla-lee-410053a8/'
         ]

## initialize empty list
names = []
jobs = []
company = []
education = []
location = []
school = []

# **Step 3: finding `xpath`** and using skipping-time
# For example, you can find the infomation of the `job title` by searching the `inspect` then copy its `full xpath`
## append the obtained values from Selector function
T0 = time.time()
for k in range(len(links)):
    t0 = time.time()
    path = links[k]
    driver.get(path)
    sel = Selector(text=driver.page_source)     
    names.append(sel.xpath("/html/body/div[7]/div[3]/div/div/div/div/div[2]/main/div[1]/section/div[2]/div[2]/div[1]/ul[1]/li[1]/text()").extract_first())
    jobs.append(  sel.xpath("/html/body/div[7]/div[3]/div/div/div/div/div[2]/main/div[1]/section/div[2]/div[2]/div[1]/h2/text()").extract_first() )
    company.append(sel.xpath('/html/body/div[7]/div[3]/div/div/div/div/div[2]/main/div[1]/section/div[2]/div[2]/div[2]/ul/li[1]/a/span/text()').extract_first())
    education.append(sel.xpath('/html/body/div[7]/div[3]/div/div/div/div/div[2]/main/div[2]/div[5]/span/div/section/div[2]/section/ul/li[1]/div/div/a/div[2]/div/p[1]/span[2]/text()').extract_first())
    location.append(sel.xpath('/html/body/div[7]/div[3]/div/div/div/div/div[2]/main/div[1]/section/div[2]/div[2]/div[1]/ul[2]/li[1]/text()').extract_first())
    school.append(sel.xpath('/html/body/div[7]/div[3]/div/div/div/div/div[2]/main/div[2]/div[5]/span/div/section/div[2]/section/ul/li[1]/div/div/a/div[2]/div/h3/text()').extract_first())
    print("step = %s, appending complete after %s (seconds) "%(k, time.time() - t0))
    if (k+1) % 10 == 0:
        t0 = time.time()
        wait_time = 50   ## timeout variable [seconds] can be omitted, if you use specific value in the while condition
        while time.time() < t0 + wait_time:
            stop = 0
            if stop == 50:
                break
            stop -= 1
        print('stoping_time = ',time.time() - t0)
print('Attaching progress : Completed!!! \t total url_links = %s, total_time = %s'%(len(links), time.time() - T0)))

## Step 4. Create dataframe
import pandas as pd
df = pd.DataFrame({'name': names, 
                   'job title': jobs, 
                   'company': company, 
                   'education': education, 
                   'school': school, 
                   'location': location, 
                   'url': links}) 
df
## Text-processing
def remove(x):
    if x == None:
        x = None
    else:
        x = x.replace('\n\t', '').replace('\n', '')
    return x
for col in df.columns:
    df[col] = df[col].apply(lambda x : remove(x))
df

# ## 3. Save & verifying saved-`csv.file`
#df.to_csv('linkedin.csv')

## Display saved-file
#pd.read_csv('linkedin.csv')