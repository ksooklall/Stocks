import os
import pandas as pd

import requests
from bs4 import BeautifulSoup as bs


def get_list(ticker):

	base_url = 'http://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&CIK={}&type=&dateb=&owner=&start={}&count=100&output=xml'
	
	href = []
	
	for page_number in range(0, 2000, 100):

		pg_url = base_url.format(ticker, page_number)
		sec_page = requests.get(pg_url)
		sec_soup = bs(str(sec_page.text))
		
		filings = sec_soup.findAll('filing')

		for filing in filings:
			report_year = int(filing.datefiled.get_text()[0:4])
			if (filing.type.get_text() == "10-Q") & (report_year > 2017):
				print(filing.filinghref.get_text())
				href.append(filing.filinghref.get_text())
	
	return href

def download_report(url_list, dir_path):
	
	sec_url = 'http://www.sec.gov'
	target_file_type = u'EX-101.INS'
	
	for report_url in url_list:
		report_page = requests.get(report_url)
		report_soup = bs(str(report_page.text))
		
		xbrl_file = report_soup.findAll('tr')
		
		for item in xbrl_file:
			try:
				if item.findAll('td')[3].get_text() == target_file_type:
					if not os.path.exists(dir_path):
						os.makedirs(dir_path)
							 
					target_url = sec_url + item.findAll('td')[2].find('a')['href']
					print('Target URL found!')
					print('Target URL is: {}'.format(target_url))
					
					file_name = target_url.split('/')[-1]
					print(file_name)
				   
					xbrl_report = requests.get(target_url)
					
					with open(os.path.join(dir_path,file_name), 'w+') as f:
						f.write(xbrl_report.text)
			except:
				pass


tickers = ['SNAP']

for ticker in tickers:
	url_list= get_list(ticker)
	base_path = 'filings/'
	dir_path = base_path +ticker
	download_report(url_list, dir_path)

