# Demo notes (about 4 minutes)

**Before you start:** run `streamlit run app/streamlit_app.py` and open the app. Click each example
once so the model is warmed up. Keep a second tab open on `?example=suspicious` as a backup.

1. **Problem (30 s).** Buyers can't tell whether a listing is fairly priced. Similar flats in the
   same area are listed at very different prices, and fake or inflated listings are common.
2. **Data and model (45 s).**
   - 13,320 Kaggle listings from Bengaluru, cleaned to 11,740.
   - Features: locality, area, BHK, bathrooms, balconies, area type, construction status.
   - Gradient boosting on log(price). Typical error is about 22%, and the main model is roughly tied
     with a linear baseline (say so plainly).
3. **Live demo (2 min).**
   - **Try a fair listing** (Whitefield 2 BHK, ₹68L): verdict is Fairly Priced, the estimate is ₹65L,
     and the asking line sits inside the shaded fair range.
   - **Try an overpriced listing** (Electronic City 2 BHK, ₹75L): 58% above the estimate, to the
     right of almost all similar listings. Locality is one of the factors lowering the estimate.
   - **Try a suspicious listing** (Koramangala 3 BHK, ₹55L): 65% below the estimate, too good to be
     true. Locality and area are the biggest factors raising it.
   - **Custom listing:** change the area or asking price in the sidebar and click **Check price**.
4. **Limits and next steps (30 s).**
   - Limits: no photos, exact location, floor or age; prices are from around 2017–2020; asking
     prices, not sale prices.
   - Next steps: a Spark pipeline for larger data, more cities, duplicate-listing detection.

## Likely questions

- **Why isn't price per sq ft an input?** It's calculated from the price, which is what we're
  predicting. Using it would be leakage: near-perfect accuracy in testing, useless for a new
  listing. We only use it during cleaning, to drop extreme outliers.
- **How were the thresholds chosen?** We looked at actual ÷ predicted prices on validation listings
  the model hadn't seen. The middle 80% (0.75–1.40) is "Fairly Priced", and the outer 2.5% at each
  end is "Suspicious". On the test set, 77% of real listings come out fair.
- **Why not just use linear regression?** It's about as accurate here, and we report that honestly.
  Gradient boosting is slightly better on percentage error, SHAP explains it exactly, and it can
  capture interactions such as plots in expensive localities.
- **Isn't encoding locality with prices leakage?** No. It's learned only from training data, with
  cross-fitting so a listing never sees its own price. Test listings use only what was learned
  during training.
- **How do the explanations work?** SHAP splits the log-price prediction exactly into one
  contribution per feature, compared with a typical listing (about ₹79L). We convert each
  contribution to rupees.
