# Which Recalls Do Owner Complaints Actually Warn Us About?

**MSDS 692 Data Science Practicum · Apurba Poudel · Regis University**

## What this project is about

When something goes wrong with a car, owners can file a complaint with NHTSA (the National Highway Traffic Safety Administration). Sometimes a recall follows. This project asks a simple question: **how often do owner complaints show up before a recall, and how early?**

The audience is the screening staff at NHTSA's Office of Defects Investigation, who read incoming complaints and decide which vehicles to look into.

## The short answer

**NHTSA should investigate as soon as complaints pile up on one vehicle. Most recalls come with no warning, but when owners do warn, it's about two years early.**

- Only about **8%** of defect recalls (229 of 2,776, from 2016 to 2019) had an owner complaint first.
- For recalls NHTSA had to push, it was **19%**. For recalls carmakers started on their own, it was **7%**.
- When complaints did come first, the recall followed about **two years** later.
- More early complaints meant a faster recall.
- A simple monthly list (vehicles with 10 or more complaints in six months) flagged about 35 vehicles a month. Those vehicles were **3.6 times** as likely to be recalled, a median of 17 months ahead.

## Data

Three public files from NHTSA, downloaded from <https://www-odi.nhtsa.dot.gov/downloads/flatfiles.cfm>:

| File | What's in it | Rows |
|---|---|---|
| `COMPLAINTS_RECEIVED_2010-2014.txt` | owner complaints, 2010–2014 | 392,970 |
| `COMPLAINTS_RECEIVED_2015-2019.txt` | owner complaints, 2015–2019 | 489,809 |
| `FLAT_RCL_POST_2010.txt` | recalls, 2010 onward | 245,008 |

The files are about 940 MB together, so they're not in this repo. Download and unzip them into one folder. All the rows are used; nothing is sampled.

## Method in plain words

1. Group complaints into one row per vehicle and part (make, model, model year, component). That gives 133,053 cases.
2. Use only each case's first six months of complaints, so the model can't peek ahead.
3. Use survival analysis (Cox models) to measure how fast recalls come. I fit one model for company recalls and one for NHTSA-pushed recalls. More than 98% of cases have no recall yet, and this method treats them as "still waiting" instead of "never."
4. As a comparison, I also tried yes/no models (logistic regression, random forest, gradient boosting, and a text model). The best was only 2.8 times better than guessing, so I set them aside.
5. Two extra checks: whether the warning rate holds when late complaints are dropped (it does), and a backtest of the monthly list.

## How to run it

You need Python 3.10 or newer and about 8 GB of free memory.

```bash
pip install -r requirements.txt
```

**Notebook (easiest):** open `recall_warning_analysis.ipynb`, check that `DATA_DIR` in the Settings cell points to your data folder, then run all cells. The last cell checks the main numbers and prints OK next to each one.

**Script:**

```bash
python run_all.py            # full run, about 10 minutes
python run_all.py --fast     # quicker run without the slow steps
```

Both read the data from the folder set in `DATA_DIR` (currently `/Users/oxy/Desktop/MSDS/prac 1/Data`). To use a different folder, change `DATA_DIR` or pass the folder: `python run_all.py "path/to/folder"`.

Results go to `output/`: `results.json` with every number, plus the charts.

**Slides (optional, needs Node.js):**

```bash
cd deck
npm install
npm run build                # writes output/Final_Deck.pptx
```

## What's in the repo

```
recall_warning_analysis.ipynb   the full analysis, step by step
run_all.py                      the same analysis as one script
run_checks.py                   only the two extra checks (fast)
src/                            the code the notebook and script use
  load.py        reads the NHTSA files
  features.py    builds the cases
  warning.py     how often complaints came before a recall
  timing.py      time from first complaint to recall
  models.py      the Cox models
  classify.py    the yes/no models I tested
  checks.py      the late-complaint check and monthly-list backtest
  cases.py       the 2012 Kia Soul example
  charts.py      the charts
deck/                           builds the presentation
requirements.txt                Python packages
```

## Limitations

- These are patterns, not proof that complaints cause recalls.
- Complaints are voluntary, so no complaint doesn't mean no problem.
- Model names are written differently in the two files, so some recalls can't be matched to their complaints.
- The data ends in 2019, so the monthly list hasn't been tested on newer years.
