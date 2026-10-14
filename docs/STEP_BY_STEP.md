# WaveWash: step by step, from download to published

Everything below is for a Mac. Replace `BoobalanArjunan` with your own.

## Part A. Put it on your computer and try it

1. **Find the download.** Files you downloaded are in your `Downloads` folder.
   You need `wavewash-source.zip`.
2. **Unzip it.** Double-click the zip. You get a folder called `wavewash-tool`.
3. **Move the folder** somewhere permanent, for example `Documents`.
4. **Open Terminal** (press Cmd+Space, type Terminal, press Enter).
5. **Go into the folder:**
   `cd ~/Documents/wavewash-tool`
6. **Make a private Python space** (done once):
   `python3 -m venv .venv`
   then switch it on (every time you open a new Terminal):
   `source .venv/bin/activate`
7. **Install WaveWash and the test tools:**
   `pip install -e ".[dev]"`
8. **Run the tests.** This is the "does it work on my Mac?" check:
   `python -m pytest -q`
   You want to see `23 passed`. If something fails, copy the message and send it to me.
9. **Try the app:** `wavewash app`. Your browser opens. Choose an input folder with .wav files and
   an output folder that is NOT inside the input folder, then click Run.
10. **Or use the command line:**
    `wavewash run --input ~/Desktop/my_wavs --output ~/Desktop/my_wavs_clean/clean --purpose asr`

## Part B. Test it on real speech (the most important step)

11. Download a free sample of real speech, for example LibriSpeech **dev-clean** from openslr.org/12.
12. Run WaveWash on a few hundred files (step 10 with that folder).
13. Open the results. Look at the first 50 files it rejected and ask: *was it really bad?*
    Write down: files checked, files rejected, how many of the rejects were really bad.
    These real numbers go in the README. If many good files were rejected, tell me and we fix the settings.

## Part C. Put the code on GitHub

14. Make a free account at github.com.
15. Click **New repository**. Name it `wavewash`. Choose Public. Do not tick any extra files.
16. In Terminal, inside `wavewash-tool`:
    ```
    git init
    git add .
    git commit -m "WaveWash 0.1.0a1"
    git branch -M main
    git remote add origin https://github.com/BoobalanArjunan/wavewash.git
    git push -u origin main
    ```
17. In the README, replace `BoobalanArjunan` in the badge line.
18. On GitHub open the **Actions** tab. The tests now run by themselves on Mac, Windows and Linux.
    Green tick means good. Red means something broke on that system, and we fix it.

## Part D. Publish to PyPI (so anyone can `pip install wavewash`)

18b. **Put your name on it.** Run this once, with your own details (the email is optional and becomes PUBLIC):
    `python scripts/set_owner.py --name "Your Full Name" --github your-github-name`
    It sets you as author and maintainer, fixes the project links and the licence holder, and replaces every
    `BoobalanArjunan`. On PyPI you are the *owner* of the project because you upload it from your PyPI account.

19. Check the name is free: open pypi.org/project/wavewash. If the page says "not found", it is free.
20. Make accounts at **test.pypi.org** and **pypi.org** (turn on two-factor login on both).
    On each, create an **API token** (Account settings, API tokens).
21. Build the files (inside `wavewash-tool`, with the venv switched on):
    ```
    pip install build twine
    python -m build
    ```
    The files appear in the `dist` folder.
22. **Practice upload first** (uses the test site; username is `__token__`, password is your token):
    `python -m twine upload --repository testpypi dist/*`
23. **Check it installs** in a fresh space:
    ```
    python3 -m venv ~/try-wavewash && source ~/try-wavewash/bin/activate
    pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple wavewash
    wavewash doctor
    ```
24. **Real upload.** You can't undo it, so only after step 23 works:
    `python -m twine upload dist/*`
25. Anyone can now run `pip install wavewash`.

## What each part is for
- **Tests** protect users. A change that breaks something shows up before it reaches them.
- **Real-speech numbers** show it works on real data.
- **GitHub and the green tick** give people a reason to trust it.
- **TestPyPI first** because a mistake on real PyPI can't be erased.
