# Fair Price Checker

Is this asking price fair? Enter a property's details and asking price to get an
estimated fair price, a verdict (Good Deal / Fairly Priced / Overpriced / Suspicious),
the factors behind the estimate, and how the listing compares with similar properties.

*Work in progress.*

## How to run

```bash
pip install -r requirements.txt
python src/train.py
streamlit run app/streamlit_app.py
```

## Dataset

[Bengaluru House Price Data](https://www.kaggle.com/datasets/amitabhajoy/bengaluru-house-price-data) (Kaggle),
about 13,300 listings across Bengaluru. Place `Bengaluru_House_Data.csv` in `data/raw/` (not committed).

## Team

- Advika Awasthi
- Utkarsh Mallik
