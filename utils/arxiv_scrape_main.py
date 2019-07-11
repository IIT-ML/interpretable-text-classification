"""
    arxiv script
"""

import os
import argparse
import pandas as pd
import numpy as np
from arxivscraper import *
import logging

os.environ['TZ'] = 'America/Chicago'
logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s')

def main():
    parser = argparse.ArgumentParser(description="Interpretable Cautious Text Classifier args")
    parser.add_argument('--year', default=2018, type=int, help="If given, the operation will be operated in GPU")
    args = parser.parse_args()

    start = ['{}-{:02d}-01'.format(args.year,i) for i in range(1,13)]

    end = ['{}-01-31'.format(args.year), '{}-02-28'.format(args.year), 
             '{}-03-31'.format(args.year), '{}-04-30'.format(args.year),
             '{}-05-31'.format(args.year), '{}-06-30'.format(args.year),
             '{}-07-31'.format(args.year), '{}-08-31'.format(args.year),
             '{}-09-30'.format(args.year), '{}-10-31'.format(args.year),
             '{}-11-30'.format(args.year), '{}-12-31'.format(args.year)]
    cols = ('id', 'title', 'categories', 'abstract', 'doi', 'created', 'updated', 'authors')
    if not os.path.exists('../dataset/arxiv/{}'.format(args.year)):
        os.mkdir('../dataset/arxiv/{}'.format(args.year))
    for st, en in zip(start, end):
        print('Processing...', st, ' - ', en)
        scraper = Scraper(category='cs', date_from=st,date_until=en, t=50)
        output = scraper.scrape()
        df = pd.DataFrame(output, columns=cols)
        df.to_csv('../dataset/arxiv/{}/cs-{}-{}.csv'.format(args.year, st.split('-')[0], st.split('-')[1]), encoding='utf-8', index=False)
    
if __name__ == '__main__':
    main()
