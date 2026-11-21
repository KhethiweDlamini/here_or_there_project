# Did I lose it here, or there?

A last-seen object finder for low-resource settings. A phone on a stand takes a photo of a room every 30 seconds. When you ask where an object is, it says which of 8 zones the object was last seen in, and when, or that it has not seen it.

**Read the write-up:** <https://khethiwedlamini.github.io/here_or_there_project/>

**Team:** Christiana Adisa, Safia Gibril, Khethiwe Dlamini and Alaa Almouiz Farog Mohammed, AIMS, November 2026.

## Results in brief

On 50 held-out test questions (Session 2), our final system, a YOLO11-s detector fine-tuned on Session 1, named the right zone 48% of the time. Random guessing and a "usual place" rule both got 12%. It found the water bottle and mug almost every time, never found the keys or wallet, and never reported an object that was not there. The write-up has the full results, failure cases and limitations.

## What is in this repository

### The write-up website

| Path | What it is |
| --- | --- |
| `index.html` | The write-up as a single-page blog post: problem, method, results, an interactive demo and limitations. All styles, scripts and the demo data are inside this file. |
| `video.mp4`, `poster.jpg` | Short video (and its preview frame) showing how the app should work. |
| `room.jpg` | The test room as the fixed camera sees it. The page draws the 8 zones on top of it. |
| `keys.png` | Close-up of the keys, used by the "keys at the size each model receives them" figure. |
| `f/` | The 60 Session 2 test photos (`IMG001.jpg` to `IMG060.jpg`), reduced to 540 × 720, used by the demo. People are blurred. |
| `reg/` | 25 registration photos (5 per object) shown in the post. These are examples, not the full set of 207. |
| `.nojekyll` | Tells GitHub Pages to serve the files as they are. |

### Experiments

| Path | What it is |
| --- | --- |
| `notebooks/here_or_there_YOLO.ipynb` | The notebook for the **final system**: YOLO11-s fine-tuned on Session 1, run on 640-pixel tiles of each photo, with a confidence threshold of 0.63 chosen on Session 1. These are the results reported on the website. |
| `notebooks/here_or_there_pre-trained_models.ipynb` | The Kaggle/Colab notebook for the **zero-training methods** we tried first: DINOv2 zone crops (round 1), DINOv2 tiles and OWLv2 (round 2), and OWLv2 with the usual-place prior (round 3). It enforces the test protocol in code: settings are chosen on Session 1 only, and the Session 2 questions are fixed before any Session 2 photo is processed. Setup steps are in the notebook's first cell. |
| `plots/make_plots.py` | Draws five slide-ready charts. The numbers are copied into the script from the pre-trained models notebook's round 3 run (Kaggle Version 5). |
| `plots/*.png` | The five charts: Session 1 vs Session 2 gap, methods vs chance, answer breakdown, zones where objects were found, and the H1/H2 hypothesis checks. |
| `plots/fonts/` | Lato and Noto Serif fonts used by the charts (SIL Open Font License). |

**Note:** `plots/` shows the round 3 OWLv2 results from the pre-trained models notebook, not the final YOLO11-s results. The YOLO11-s numbers are on the website.

### Not in this repository

The full photo sessions, all 207 registration photos, the placement log (`placement_log_v4_corrected_with_S2.xlsx`) and the Roboflow labels are kept in a private Kaggle dataset, because people appear in the photos. The notebooks need these files to run.

## How to use it

**View the write-up locally:** open `index.html` in a browser. Everything except the fonts and voice questions works offline.

**Re-run the zero-training experiments:** follow the setup in the first cell of `notebooks/here_or_there_pre-trained_models.ipynb`. In short, upload the data to a private Kaggle dataset, import the notebook, attach the dataset, turn Internet on (to download DINOv2 and OWLv2) and run all. Set `ROUND` in cell A1 to 1, 2 or 3. Results are written to a `results/` folder, including a readable `report_r3.html`.

**Redraw the charts:** if you re-run the notebook, update `DATA` at the top of `plots/make_plots.py`, then run:

```
pip install numpy matplotlib
python plots/make_plots.py
```

## How we used Claude

We used Claude, Anthropic's AI assistant, throughout the project. It helped with the proposal and slides, drafted the pre-trained models notebook and later added rounds 2 and 3, the results report and fixes, wrote the plotting script, and built the write-up page. It also reviewed and fixed the team's edits to the page and wrote this README. Commits where Claude wrote code are marked `Co-Authored-By: Claude` in the git history. The full statement is in the Sources section of the write-up.

## Notes

- Voice questions in the demo use the browser's speech recognition. They work in Chrome and Edge and need an internet connection. Spoken answers use the browser's own voice.
- The demo answers come from our placement log, using the same last-seen rule as the system. They show what the system should say, and each answer shows how often the real system got that object right in the test.
- People in the photos are blurred, and so are the certificates in the glass cabinet.
