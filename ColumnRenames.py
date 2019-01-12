"""
This file contains columns renames from sites that have been scraped
"""

form4 = {
    'averageprice': 'avg_price',
    'insiderrelationship': 'insider_pos',
    'sharesowned': 'shares_owned',
    'sharestraded': 'shares_traded',
    'transactiondate': 'date'
    }

nasdaq = {'CompanyName(Symbol)MarketCapSortby:Name/Size': 'sym_mc_size',
          'ExpectedReportDate': 'expected_date',
          'ReportDate': 'expected_date',
          "LastYear'sReportDate": 'last_yr_report_date',
          "LastYear'sEPS*": 'last_yr_eps',
          'ConsensusEPS*Forecast': 'consensus_eps',
          '%Suprise': 'suprise_pct',
          '#ofEsts': 'analysts',
          'FiscalQuarterEnding': 'quarter_ending'}
