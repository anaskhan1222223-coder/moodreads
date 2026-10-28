"""
MoodReads 2.0 — mood-aware book discovery.
Face mood · Voice search · Mood & genre browsing · TF-IDF AI matching · Buy links · Reading list
"""
import datetime as dt
import html as _html
import random
import re
import tempfile
import urllib.parse
import zlib
from pathlib import Path

import pandas as pd
import requests
import streamlit as st
import streamlit.components.v1 as components
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(
    page_title="MoodReads — the book that fits how you feel",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="collapsed",
)

AMAZON_DOMAIN = "amazon.com"  # change to "amazon.in" for India


# ════════════════════════════════════════════════════════════════
# HELPERS
# ════════════════════════════════════════════════════════════════
def esc(s):
    return _html.escape(str(s), quote=True)


def squash(s, sep=""):
    """Collapse multi-line markup so Streamlit's markdown never treats it as code."""
    return sep.join(line.strip() for line in s.strip().splitlines() if line.strip())


def norm(s):
    return re.sub(r"[^a-z0-9]+", " ", str(s).lower()).strip()


def hue_of(s):
    return zlib.crc32(str(s).encode()) % 360


def book_key(b):
    return f"{norm(b['title'])}|{norm(b['author'])}"


# ════════════════════════════════════════════════════════════════
# DESIGN SYSTEM (CSS)
# ════════════════════════════════════════════════════════════════
CSS = """
@import url('https://fonts.googleapis.com/css2?family=Fraunces:ital,opsz,wght@0,9..144,500;0,9..144,700;1,9..144,500&family=Manrope:wght@400;500;600;700;800&display=swap');
:root{--bg:#0B0B0F;--s1:#13131A;--s2:#1B1B24;--line:rgba(255,255,255,.08);--line2:rgba(255,255,255,.15);
--text:#F4F1EA;--muted:#9C99A3;--accent:#FF6A3D;--accent2:#FFB36B;--ease:cubic-bezier(.2,.7,.2,1)}
html,body,.stApp,button,input,textarea{font-family:'Manrope',system-ui,-apple-system,'Segoe UI',Roboto,sans-serif}
.stApp{background:var(--bg);color:var(--text);isolation:isolate}
.stApp::before{content:"";position:fixed;inset:0;z-index:-1;pointer-events:none;
background:radial-gradient(60vw 50vh at 85% -10%,rgba(255,106,61,.17),transparent 60%),
radial-gradient(50vw 50vh at -10% 25%,rgba(110,100,255,.10),transparent 60%)}
[data-testid="stAppViewContainer"],[data-testid="stMain"]{background:transparent!important}
#MainMenu,footer,[data-testid="stToolbar"],[data-testid="stDecoration"],[data-testid="stHeader"],[data-testid="stStatusWidget"]{display:none!important}
.block-container{max-width:1240px;padding:1.4rem 1.25rem 4rem!important}
@media(max-width:640px){.block-container{padding:.9rem .8rem 3rem!important}}
::selection{background:var(--accent);color:#16110F}

/* tabs */
[data-baseweb="tab-list"]{gap:4px;overflow-x:auto;scrollbar-width:none;border-bottom:1px solid var(--line)}
[data-baseweb="tab-list"]::-webkit-scrollbar{display:none}
button[data-baseweb="tab"]{height:auto;padding:.75rem 1rem;background:transparent;white-space:nowrap}
button[data-baseweb="tab"] p{font-size:.92rem;font-weight:700;color:var(--muted)!important;transition:color .25s}
button[data-baseweb="tab"]:hover p,button[data-baseweb="tab"][aria-selected="true"] p{color:var(--text)!important}
[data-baseweb="tab-highlight"]{background:var(--accent)!important;height:2px!important}
[data-baseweb="tab-border"]{background:transparent!important}

/* buttons */
.stButton>button,.stDownloadButton>button,.stFormSubmitButton>button{width:100%;border-radius:999px;border:1px solid var(--line2);
background:rgba(255,255,255,.035);color:var(--text);font-weight:700;padding:.55rem 1.1rem;
transition:transform .3s var(--ease),background .25s,border-color .25s,box-shadow .3s}
.stButton>button p,.stDownloadButton>button p,.stFormSubmitButton>button p{color:inherit!important;font-weight:700}
.stButton>button:hover,.stDownloadButton>button:hover,.stFormSubmitButton>button:hover{border-color:var(--accent);transform:translateY(-2px);box-shadow:0 10px 24px -12px rgba(255,106,61,.6)}
.stButton>button:active{transform:scale(.98)}
button[kind^="primary"],button[data-testid^="stBaseButton-primary"]{background:var(--accent)!important;border-color:var(--accent)!important;color:#16110F!important}
button[data-testid^="stBaseButton-pills"]{border-radius:999px;background:var(--s1);border:1px solid var(--line2);color:var(--text);transition:transform .25s var(--ease),border-color .25s}
button[data-testid^="stBaseButton-pills"]:hover{border-color:var(--accent);transform:translateY(-1px)}
button[data-testid="stBaseButton-pillsActive"]{background:var(--accent)!important;border-color:var(--accent)!important;color:#16110F!important}
button[data-testid^="stBaseButton-pills"] p{color:inherit!important;font-weight:700}

/* inputs */
.stTextInput input,.stTextArea textarea{background:var(--s1)!important;color:var(--text)!important;border-radius:14px!important}
[data-baseweb="input"],[data-baseweb="textarea"],[data-baseweb="select"]>div{background:var(--s1)!important;border-color:var(--line2)!important;border-radius:14px!important}
[data-baseweb="select"] *{color:var(--text)!important}
[data-testid="stWidgetLabel"] p,label p{color:var(--muted)!important;font-weight:600}
[data-testid="stForm"]{border:0!important;padding:0!important}
[data-testid="stAlert"]{border-radius:16px}

/* hero */
.hero{display:grid;grid-template-columns:1.05fr .95fr;align-items:center;gap:1.5rem;padding:1.8rem 0 .6rem}
.eyebrow{display:inline-flex;gap:.55rem;align-items:center;padding:.42rem .85rem;border:1px solid var(--line2);border-radius:999px;
font-size:.7rem;letter-spacing:.15em;text-transform:uppercase;color:var(--muted);font-weight:800}
.eyebrow b{width:7px;height:7px;border-radius:50%;background:var(--accent);animation:pulse 2.2s infinite}
.display{font-family:'Fraunces',Georgia,serif;font-weight:500;font-size:clamp(2.5rem,6.4vw,5.1rem);line-height:1.02;letter-spacing:-.025em;margin:1.1rem 0 1rem;color:var(--text)}
.display em{font-style:italic;color:var(--accent)}
.lead{max-width:34rem;font-size:clamp(1rem,1.5vw,1.14rem);color:var(--muted);line-height:1.7}
.stats{display:grid;grid-template-columns:repeat(4,1fr);gap:10px;margin-top:1.8rem;max-width:34rem}
.stat{padding:.85rem 1rem;border:1px solid var(--line);border-radius:16px;background:rgba(255,255,255,.025)}
.stat b{font-family:'Fraunces',serif;font-size:1.7rem;font-weight:700;display:block;line-height:1.1}
.stat span{color:var(--muted);font-size:.72rem;font-weight:600}
@media(max-width:900px){.hero{grid-template-columns:1fr}.stats{max-width:none}}
@media(max-width:520px){.stats{grid-template-columns:repeat(2,1fr)}}

/* 3D ring */
.scene{perspective:1300px;height:clamp(300px,42vw,450px);display:flex;align-items:center;justify-content:center;--w:clamp(104px,14vw,172px);animation:float 8s ease-in-out infinite}
.ring{position:relative;width:var(--w);height:calc(var(--w)*1.5);transform-style:preserve-3d;animation:spin 34s linear infinite;will-change:transform}
.it{position:absolute;inset:0;border-radius:4px 10px 10px 4px;overflow:hidden;backface-visibility:hidden;-webkit-backface-visibility:hidden;
transform:rotateY(calc(var(--k)*60deg)) translateZ(calc(var(--w)*1.18));box-shadow:0 24px 40px -18px rgba(0,0,0,.8)}
.it img{position:absolute;inset:0;width:100%;height:100%;object-fit:cover}
.it::after,.bk-cover::after{content:"";position:absolute;inset:0;pointer-events:none;
background:linear-gradient(90deg,rgba(0,0,0,.38),rgba(255,255,255,.2) 4%,transparent 10%,transparent 90%,rgba(0,0,0,.2))}
.ph{position:absolute;inset:0;display:grid;place-items:center;padding:12px;text-align:center;
background:linear-gradient(155deg,hsl(var(--h),55%,32%),hsl(var(--h2),60%,14%))}
.ph span{font-family:'Fraunces',serif;font-weight:700;font-size:.95rem;line-height:1.15;color:rgba(255,255,255,.93)}
@keyframes spin{from{transform:rotateX(-9deg) rotateY(0)}to{transform:rotateX(-9deg) rotateY(360deg)}}
@keyframes float{0%,100%{transform:translateY(0)}50%{transform:translateY(-10px)}}
@keyframes pulse{0%{box-shadow:0 0 0 0 rgba(255,106,61,.6)}70%{box-shadow:0 0 0 9px rgba(255,106,61,0)}100%{box-shadow:0 0 0 0 rgba(255,106,61,0)}}
@keyframes rise{from{opacity:0;transform:translateY(22px)}to{opacity:1;transform:none}}

/* sections */
.sec{margin:2.4rem 0 1.1rem}
.sec .k{font-size:.72rem;letter-spacing:.17em;text-transform:uppercase;color:var(--accent);font-weight:800}
.sec .t{font-family:'Fraunces',serif;font-weight:500;font-size:clamp(1.7rem,3.6vw,2.6rem);line-height:1.1;margin-top:.4rem;letter-spacing:-.01em}
.sec .s{color:var(--muted);max-width:38rem;font-size:.97rem;margin-top:.55rem;line-height:1.6}

/* book card */
.bk{background:linear-gradient(180deg,var(--s2),var(--s1));border:1px solid var(--line);border-radius:22px;padding:12px 14px 16px;margin-bottom:.5rem;
animation:rise .7s var(--ease) both;animation-delay:calc(var(--i,0)*50ms);transition:border-color .3s}
.bk-stage{position:relative;perspective:900px;padding:18px 10px 20px;display:flex;justify-content:center;border-radius:16px;
background:radial-gradient(70% 60% at 50% 58%,rgba(255,255,255,.07),transparent 72%)}
.bk-cover{position:relative;width:58%;max-width:176px;aspect-ratio:2/3;border-radius:4px 10px 10px 4px;overflow:hidden;
box-shadow:0 20px 30px -12px rgba(0,0,0,.75),0 2px 6px rgba(0,0,0,.5);transform:rotateY(-16deg) rotateX(2deg);transform-origin:left center;
transition:transform .6s var(--ease),box-shadow .6s var(--ease);will-change:transform}
.bk-cover img{position:absolute;inset:0;width:100%;height:100%;object-fit:cover}
@media(hover:hover) and (pointer:fine){.bk:hover{border-color:var(--line2)}
.bk:hover .bk-cover{transform:rotateY(-3deg) rotateX(0) translateY(-8px) scale(1.04);box-shadow:0 34px 40px -16px rgba(0,0,0,.85)}}
.match{position:absolute;top:10px;right:10px;z-index:2;font-size:.68rem;font-weight:800;padding:.3rem .6rem;border-radius:999px;background:var(--accent);color:#16110F}
.bk-tags{display:flex;gap:6px;flex-wrap:wrap;margin:12px 0 8px}
.tag{font-size:.64rem;font-weight:800;letter-spacing:.07em;text-transform:uppercase;padding:.28rem .6rem;border-radius:999px;border:1px solid var(--line2);color:var(--muted)}
.tag.m{color:var(--accent2);border-color:rgba(255,179,107,.35)}
.bk-t{font-family:'Fraunces',serif;font-size:1.17rem;line-height:1.22;font-weight:700;min-height:2.44em;display:-webkit-box;-webkit-line-clamp:2;-webkit-box-orient:vertical;overflow:hidden;color:var(--text)}
.bk-a{color:var(--muted);font-size:.84rem;margin:.35rem 0 .45rem;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.bk-r{font-size:.82rem;font-weight:700;color:var(--accent2);margin-bottom:.35rem}
.bk-r i{font-style:normal;color:var(--muted);font-weight:500}
.bk-d{color:#B7B4BD;font-size:.85rem;line-height:1.55;display:-webkit-box;-webkit-line-clamp:3;-webkit-box-orient:vertical;overflow:hidden;min-height:4.65em}
.bk-w{font-size:.76rem;color:var(--accent2);margin-top:.5rem;font-weight:600}
.buy{display:flex;flex-wrap:wrap;gap:6px;margin-top:12px}
.buy a{font-size:.74rem;font-weight:700;padding:.42rem .8rem;border-radius:999px;background:rgba(255,255,255,.06);color:var(--text)!important;
text-decoration:none!important;border:1px solid var(--line);transition:background .25s,transform .3s var(--ease),color .25s}
.buy a:hover{background:var(--accent);color:#16110F!important;transform:translateY(-2px)}

/* panels */
.empty{border:1px dashed var(--line2);border-radius:20px;padding:2.4rem 1.2rem;text-align:center;color:var(--muted);line-height:1.6}
.empty b{display:block;font-family:'Fraunces',serif;font-size:1.3rem;color:var(--text);margin-bottom:.3rem;font-weight:500}
.panel{border:1px solid var(--line);border-radius:22px;padding:1.2rem 1.3rem;background:linear-gradient(160deg,rgba(255,255,255,.05),rgba(255,255,255,.015));color:var(--muted);font-size:.92rem;line-height:1.65}
.panel b.h{display:block;color:var(--text);font-family:'Fraunces',serif;font-size:1.25rem;font-weight:500;margin-bottom:.4rem}
.panel ol{margin:.4rem 0 .6rem 1.1rem;padding:0}.panel li{margin:.25rem 0}
.steps{display:grid;grid-template-columns:repeat(3,1fr);gap:14px}
@media(max-width:820px){.steps{grid-template-columns:1fr}}
.step{border:1px solid var(--line);border-radius:22px;padding:1.3rem;background:rgba(255,255,255,.025);transition:transform .45s var(--ease),border-color .3s}
.step:hover{transform:translateY(-6px);border-color:var(--accent)}
.step .n{font-family:'Fraunces',serif;font-size:2.2rem;color:var(--accent);font-style:italic}
.step b{display:block;font-size:1.05rem;margin:.2rem 0 .4rem}
.step p,.step div{color:var(--muted);font-size:.9rem;line-height:1.6}
.tw{overflow-x:auto;border:1px solid var(--line);border-radius:20px}
.cmp{width:100%;border-collapse:collapse;min-width:560px;font-size:.9rem}
.cmp th,.cmp td{padding:.85rem 1rem;text-align:left;border-bottom:1px solid var(--line);color:var(--muted)}
.cmp th{font-size:.7rem;letter-spacing:.12em;text-transform:uppercase;color:var(--text)}
.cmp td:first-child{color:var(--text);font-weight:700}.cmp td:last-child{color:var(--accent2)}
.cmp tr:last-child td{border-bottom:0}
.fm{display:grid;grid-template-columns:auto 1fr;gap:1.4rem;align-items:center;border:1px solid var(--line);border-radius:22px;padding:1.3rem;margin:1rem 0 .5rem;
background:linear-gradient(160deg,rgba(255,106,61,.12),rgba(255,255,255,.02));animation:rise .7s var(--ease) both}
@media(max-width:640px){.fm{grid-template-columns:1fr}}
.fm .big{font-family:'Fraunces',serif;font-size:clamp(2rem,5vw,3rem);line-height:1;font-style:italic;color:var(--accent)}
.fm .sub{color:var(--muted);font-size:.8rem;margin-top:.4rem;font-weight:600}
.bar{display:grid;grid-template-columns:84px 1fr 42px;gap:10px;align-items:center;font-size:.8rem;color:var(--muted);margin:.35rem 0}
.bar .trk{height:6px;border-radius:6px;background:rgba(255,255,255,.08);overflow:hidden}
.bar .trk i{display:block;height:100%;background:var(--accent);border-radius:6px}
.bar em{font-style:normal;text-align:right}
.foot{margin-top:3.5rem;padding-top:1.5rem;border-top:1px solid var(--line);color:var(--muted);font-size:.82rem;display:flex;justify-content:space-between;gap:1rem;flex-wrap:wrap}
iframe{border:0!important;color-scheme:normal}
@media(prefers-reduced-motion:reduce){*{animation:none!important;transition:none!important}}
"""
st.markdown(f"<style>{squash(CSS, ' ')}</style>", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════
# MOODS & GENRES
# ════════════════════════════════════════════════════════════════
MOOD_MAP = {
    "Fiction": ["Relaxed", "Thoughtful", "Hopeful"],
    "Thriller": ["Excited", "Intense", "Adventurous"],
    "Mystery": ["Curious", "Analytical", "Thoughtful"],
    "Romance": ["Romantic", "Hopeful", "Emotional"],
    "Fantasy": ["Imaginative", "Escapist", "Adventurous"],
    "Science Fiction": ["Curious", "Adventurous", "Analytical"],
    "Horror": ["Intense", "Excited", "Adventurous"],
    "Biography": ["Inspired", "Thoughtful", "Motivated"],
    "History": ["Curious", "Analytical", "Thoughtful"],
    "Self-Help": ["Motivated", "Inspired", "Hopeful"],
    "Young Adult": ["Adventurous", "Emotional", "Excited"],
    "Poetry": ["Emotional", "Relaxed", "Thoughtful"],
    "Non-Fiction": ["Curious", "Analytical", "Thoughtful"],
}
GENRES = list(MOOD_MAP.keys())
ALL_MOODS = ["Relaxed", "Thoughtful", "Curious", "Excited", "Adventurous", "Intense", "Romantic",
             "Hopeful", "Emotional", "Imaginative", "Escapist", "Analytical", "Inspired", "Motivated"]
MOOD_EMOJI = {"Relaxed": "🌿", "Thoughtful": "🤔", "Curious": "🔍", "Excited": "⚡", "Adventurous": "🧭",
              "Intense": "🔥", "Romantic": "💞", "Hopeful": "🌅", "Emotional": "🌧️", "Imaginative": "🪄",
              "Escapist": "🚀", "Analytical": "🧩", "Inspired": "✨", "Motivated": "💪"}
EMOTION_TO_MOOD = {"happy": "Adventurous", "sad": "Thoughtful", "angry": "Intense", "fearful": "Escapist",
                   "surprised": "Curious", "disgusted": "Analytical", "neutral": "Relaxed"}
EMOTION_LABEL = {"happy": "Happy", "sad": "Sad", "angry": "Angry", "fearful": "Fearful",
                 "surprised": "Surprised", "disgusted": "Disgusted", "neutral": "Neutral"}


def mood_label(m):
    return f"{MOOD_EMOJI.get(m, '')} {m}".strip()


# ════════════════════════════════════════════════════════════════
# CURATED CATALOGUE  (works fully offline; real ISBNs → real covers)
# ════════════════════════════════════════════════════════════════
_RAW = [
    ("The Midnight Library", "Matt Haig", "Fiction", 2020, "9780525559474", "Thoughtful, Hopeful, Emotional",
     "Between life and death sits a library of every life you could have lived. A warm, philosophical novel about regret, second chances and choosing to live."),
    ("Project Hail Mary", "Andy Weir", "Science Fiction", 2021, "9780593135204", "Curious, Adventurous, Analytical",
     "A lone astronaut wakes in deep space with no memory and must solve a science puzzle to save Earth. Clever, funny and full of problem-solving."),
    ("The Seven Husbands of Evelyn Hugo", "Taylor Jenkins Reid", "Romance", 2017, "9781501161933", "Romantic, Emotional, Escapist",
     "A reclusive Hollywood icon finally tells the glamorous, scandalous story of her seven marriages and the love of her life."),
    ("The Silent Patient", "Alex Michaelides", "Thriller", 2019, "9781250301697", "Intense, Excited, Curious",
     "A famous painter shoots her husband and never speaks again. A psychotherapist becomes obsessed with why in this twisty psychological thriller."),
    ("Educated", "Tara Westover", "Biography", 2018, "9780399590504", "Inspired, Thoughtful, Motivated",
     "A memoir of growing up in a survivalist family in rural Idaho and teaching herself enough to earn a PhD from Cambridge."),
    ("Dune", "Frank Herbert", "Science Fiction", 1965, "9780441013593", "Adventurous, Imaginative, Escapist",
     "On the desert planet Arrakis a young heir fights for survival, spice and destiny in an epic of politics, religion and ecology."),
    ("Gone Girl", "Gillian Flynn", "Thriller", 2012, "9780307588371", "Intense, Excited, Analytical",
     "On their fifth anniversary a wife disappears and her husband becomes the prime suspect in a dark, twisty marriage thriller."),
    ("Pride and Prejudice", "Jane Austen", "Romance", 1813, "9780141439518", "Romantic, Relaxed, Hopeful",
     "A witty classic about Elizabeth Bennet, class, family pressure and her tense romance with proud Mr Darcy."),
    ("The Hobbit", "J.R.R. Tolkien", "Fantasy", 1937, "9780547928227", "Adventurous, Imaginative, Escapist",
     "A comfort-loving hobbit joins dwarves and a wizard on a quest through goblin tunnels and dragon lairs to reclaim a lost treasure."),
    ("Atomic Habits", "James Clear", "Self-Help", 2018, "9780735211292", "Motivated, Inspired, Analytical",
     "A practical guide to building good habits and breaking bad ones through tiny daily changes that compound."),
    ("Sapiens", "Yuval Noah Harari", "History", 2011, "9780062316097", "Curious, Thoughtful, Analytical",
     "A sweeping history of humankind from the Stone Age to today: how cognition, agriculture, money and shared myths shaped us."),
    ("The Alchemist", "Paulo Coelho", "Fiction", 1988, "9780062315007", "Hopeful, Inspired, Relaxed",
     "A shepherd boy travels from Spain to the Egyptian desert chasing a dream in this fable about destiny and listening to your heart."),
    ("Normal People", "Sally Rooney", "Romance", 2018, "9781984822178", "Emotional, Thoughtful, Romantic",
     "Two Irish teenagers drift together and apart over the years in an intimate, quietly devastating story of love, class and communication."),
    ("The Hunger Games", "Suzanne Collins", "Young Adult", 2008, "9780439023528", "Excited, Intense, Adventurous",
     "In a dystopian nation, sixteen-year-old Katniss volunteers to fight to the death in a televised arena to save her sister."),
    ("The Name of the Wind", "Patrick Rothfuss", "Fantasy", 2007, "9780756404741", "Imaginative, Escapist, Adventurous",
     "A gifted orphan becomes the most notorious wizard of his age: a lyrical tale of magic university, music and legend."),
    ("Neuromancer", "William Gibson", "Science Fiction", 1984, "9780441569595", "Intense, Curious, Analytical",
     "A washed-up hacker is hired for one last job in cyberspace in the novel that defined cyberpunk."),
    ("It", "Stephen King", "Horror", 1986, "9781501142970", "Intense, Excited, Adventurous",
     "A shape-shifting evil awakens in the town of Derry and a group of childhood friends reunite as adults to destroy it."),
    ("Milk and Honey", "Rupi Kaur", "Poetry", 2014, "9781449474256", "Emotional, Relaxed, Hopeful",
     "Short, honest poems and illustrations about love, loss, trauma, healing and femininity."),
    ("Becoming", "Michelle Obama", "Biography", 2018, "9781524763138", "Inspired, Motivated, Hopeful",
     "The former First Lady's memoir, from childhood in Chicago to the White House, on identity, purpose and resilience."),
    ("Murder on the Orient Express", "Agatha Christie", "Mystery", 1934, "9780062693662", "Curious, Analytical, Relaxed",
     "Snowbound on a luxury train, detective Hercule Poirot must unmask a killer among the passengers in a classic whodunit."),
    ("The Song of Achilles", "Madeline Miller", "Fantasy", 2011, "9780062060624", "Emotional, Romantic, Imaginative",
     "A lyrical retelling of the Trojan War through the love between Achilles and Patroclus, full of Greek myth and heartbreak."),
    ("The Martian", "Andy Weir", "Science Fiction", 2011, "9780553418026", "Analytical, Adventurous, Motivated",
     "Stranded alone on Mars, astronaut Mark Watney uses science, duct tape and humour to survive until rescue."),
    ("Thinking, Fast and Slow", "Daniel Kahneman", "Non-Fiction", 2011, "9780374533557", "Analytical, Curious, Thoughtful",
     "A Nobel laureate explains the two systems of thought that drive our decisions, biases and judgment."),
    ("Man's Search for Meaning", "Viktor E. Frankl", "Biography", 1946, "9780807014271", "Thoughtful, Hopeful, Inspired",
     "A psychiatrist's account of surviving Nazi concentration camps and the idea that purpose can carry us through suffering."),
    ("The Girl with the Dragon Tattoo", "Stieg Larsson", "Mystery", 2005, "9780307454546", "Intense, Curious, Analytical",
     "A disgraced journalist and a brilliant hacker investigate a decades-old disappearance inside a wealthy Swedish family."),
    ("Where the Crawdads Sing", "Delia Owens", "Fiction", 2018, "9780735219090", "Thoughtful, Emotional, Relaxed",
     "A girl raised alone in the North Carolina marshes becomes the suspect in a murder in this lyrical mystery and coming-of-age story."),
    ("Anxious People", "Fredrik Backman", "Fiction", 2020, "9781501160837", "Hopeful, Relaxed, Emotional",
     "A failed bank robber takes a group of strangers hostage at an apartment viewing in a funny, tender novel about being human."),
    ("The Lightning Thief", "Rick Riordan", "Young Adult", 2005, "9780786838653", "Adventurous, Excited, Imaginative",
     "Percy Jackson discovers he is the son of Poseidon and sets off across America to stop a war among the Greek gods."),
    ("Fahrenheit 451", "Ray Bradbury", "Science Fiction", 1953, "9781451673319", "Thoughtful, Intense, Analytical",
     "In a future where books are burned, fireman Guy Montag begins to question everything."),
    ("1984", "George Orwell", "Science Fiction", 1949, "9780451524935", "Intense, Analytical, Thoughtful",
     "Surveillance, propaganda and rewritten history: Winston Smith risks everything to resist Big Brother."),
    ("Beach Read", "Emily Henry", "Romance", 2020, "9781984806734", "Romantic, Relaxed, Hopeful",
     "Two rival writers swap genres for a summer in a witty, warm romance about grief and second chances."),
    ("The Da Vinci Code", "Dan Brown", "Thriller", 2003, "9780307474278", "Excited, Curious, Adventurous",
     "A symbologist races through Paris and London solving codes hidden in art to uncover a secret society."),
    ("Mindset", "Carol S. Dweck", "Self-Help", 2006, "9780345472328", "Motivated, Inspired, Analytical",
     "A psychologist shows how a growth mindset, rather than a fixed one, shapes learning, success and resilience."),
    ("Circe", "Madeline Miller", "Fantasy", 2018, "9780316556347", "Imaginative, Escapist, Emotional",
     "The witch of Aiaia tells her own story among gods, monsters and mortals in a feminist retelling of Greek myth."),
]


def _build_catalog():
    out = []
    for t, a, g, y, isbn, moods, d in _RAW:
        out.append(dict(
            title=t, author=a, genre=g, year=str(y), isbn=isbn, description=d,
            cover=f"https://covers.openlibrary.org/b/isbn/{isbn}-M.jpg?default=false",
            moods=[m.strip() for m in moods.split(",")], rating=None, ratings_count=0, src="curated",
        ))
    return out


CATALOG = _build_catalog()
BY_TITLE = {b["title"]: b for b in CATALOG}


# ════════════════════════════════════════════════════════════════
# LIVE DATA — Open Library
# ════════════════════════════════════════════════════════════════
OL_API = "https://openlibrary.org/search.json"
OL_FIELDS = "key,title,author_name,first_publish_year,cover_i,subject,ratings_average,ratings_count,isbn,first_sentence"
HEADERS = {"User-Agent": "MoodReads/2.0 (educational project; Streamlit)"}
GENRE_KEYS = [("science fiction", "Science Fiction"), ("sci-fi", "Science Fiction"), ("fantasy", "Fantasy"),
              ("thriller", "Thriller"), ("suspense", "Thriller"), ("mystery", "Mystery"), ("detective", "Mystery"),
              ("romance", "Romance"), ("love stor", "Romance"), ("horror", "Horror"), ("biograph", "Biography"),
              ("memoir", "Biography"), ("history", "History"), ("self-help", "Self-Help"), ("self help", "Self-Help"),
              ("young adult", "Young Adult"), ("juvenile", "Young Adult"), ("poetry", "Poetry"), ("poems", "Poetry"),
              ("non-fiction", "Non-Fiction"), ("nonfiction", "Non-Fiction"), ("fiction", "Fiction")]


def pick_genre(subjects):
    blob = " | ".join(subjects or []).lower()
    for k, g in GENRE_KEYS:
        if k in blob:
            return g
    return "Non-Fiction" if blob else "Fiction"


@st.cache_data(ttl=3600, show_spinner=False)
def _fetch(query, limit, sort):
    params = {"q": query, "limit": limit, "fields": OL_FIELDS}
    if sort:
        params["sort"] = sort
    r = requests.get(OL_API, params=params, headers=HEADERS, timeout=8)
    r.raise_for_status()
    docs = r.json().get("docs", [])
    if not docs:
        raise RuntimeError("empty result")  # errors are never cached
    return docs


def _parse(docs, mood=None, require_cover=False):
    out, seen = [], set()
    for d in docs:
        title = (d.get("title") or "").strip()
        if not title or (require_cover and not d.get("cover_i")):
            continue
        k = norm(title)
        if k in seen:
            continue
        seen.add(k)
        subjects = d.get("subject") or []
        genre = pick_genre(subjects)
        subs = [s for s in subjects if len(s) < 28][:4]
        fs = d.get("first_sentence")
        fs = fs[0] if isinstance(fs, list) and fs else (fs if isinstance(fs, str) else "")
        if fs and len(fs) > 25:
            desc = "Opens with: “" + fs[:170].rstrip() + ("…" if len(fs) > 170 else "") + "”"
        elif subs:
            desc = "Themes: " + ", ".join(subs) + "."
        else:
            desc = "Open a retailer link for the full synopsis and reviews."
        ra = d.get("ratings_average")
        cover_i = d.get("cover_i")
        out.append(dict(
            title=title, author=", ".join((d.get("author_name") or ["Unknown author"])[:2]),
            genre=genre, year=str(d.get("first_publish_year") or "—"),
            isbn=(d.get("isbn") or [""])[0], description=desc,
            cover=f"https://covers.openlibrary.org/b/id/{cover_i}-M.jpg?default=false" if cover_i else "",
            moods=[mood] if mood else MOOD_MAP.get(genre, ["Curious"]),
            rating=round(float(ra), 1) if ra else None, ratings_count=int(d.get("ratings_count") or 0), src="live",
        ))
    out.sort(key=lambda b: not b["cover"])  # covers first (stable)
    return out


def live_search(query, limit=24, sort=None, mood=None, require_cover=False):
    for s in ([sort, None] if sort else [None]):
        try:
            return _parse(_fetch(query, limit, s), mood, require_cover)
        except Exception:
            continue
    return []


def dedupe(books, exclude=()):
    seen, out = set(exclude), []
    for b in books:
        k = book_key(b)
        if k not in seen:
            seen.add(k)
            out.append(b)
    return out


def search_books(q, limit=24):
    stop = {"the", "and", "book", "books", "about", "for", "with", "novel", "novels", "read", "some", "good",
            "best", "recommend", "like", "that", "this", "from", "want", "give", "show"}
    toks = [t for t in re.findall(r"[a-z0-9']+", q.lower()) if len(t) > 2 and t not in stop]
    scored = []
    for b in CATALOG:
        hay = " ".join([b["title"], b["author"], b["genre"], b["description"], " ".join(b["moods"])]).lower()
        s = sum(1 for t in toks if t in hay)
        if s and s >= (len(toks) + 1) // 2:
            scored.append((s, b))
    scored.sort(key=lambda x: -x[0])
    cur = [b for _, b in scored][:6]
    return dedupe(cur + live_search(q, limit))[:limit]


def recommend_for_mood(mood, n=12):
    cur = [dict(b, mood=mood) for b in CATALOG if mood in b["moods"]]
    cur.sort(key=lambda b: b["moods"].index(mood))
    live = []
    for g in [g for g, ms in MOOD_MAP.items() if mood in ms][:2]:
        live += live_search(f'subject:"{g.lower()}"', 16, sort="editions", mood=mood, require_cover=True)
    live = [dict(b, mood=mood) for b in live]
    return dedupe(cur[:8] + live + cur[8:])[:n]


def genre_books(genre, order):
    cur = [b for b in CATALOG if genre == "All" or b["genre"] == genre]
    q = 'subject:"fiction"' if genre == "All" else f'subject:"{genre.lower()}"'
    books = dedupe(cur + live_search(q, 20, sort="editions", require_cover=True))
    if order == "Newest first":
        books.sort(key=lambda b: int(b["year"]) if b["year"].isdigit() else 0, reverse=True)
    elif order == "Oldest first (classics)":
        books.sort(key=lambda b: int(b["year"]) if b["year"].isdigit() else 9999)
    elif order == "Title A–Z":
        books.sort(key=lambda b: b["title"].lower())
    return books[:18]


# ════════════════════════════════════════════════════════════════
# AI MODEL — TF-IDF + cosine similarity (with explanations)
# ════════════════════════════════════════════════════════════════
@st.cache_resource(show_spinner=False)
def get_ai_model():
    docs = [" ".join([b["title"], b["author"], b["genre"], b["genre"], " ".join(b["moods"]), b["description"]]) for b in CATALOG]
    vec = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), sublinear_tf=True)
    mat = vec.fit_transform(docs)
    return vec, mat


def ai_rank(qvec, exclude=None, n=6):
    vec, mat = get_ai_model()
    sims = cosine_similarity(qvec, mat).ravel()
    order = sims.argsort()[::-1]
    terms = vec.get_feature_names_out()
    picked = [i for i in order if i != exclude and sims[i] > 0][:n]
    if not picked:
        return []
    top = sims[picked[0]]
    res = []
    for i in picked:
        contrib = qvec.multiply(mat[i]).toarray().ravel()
        words = [terms[j] for j in contrib.argsort()[::-1][:3] if contrib[j] > 0]
        res.append(dict(CATALOG[i], match=min(99, int(round(sims[i] / top * 100))), why=", ".join(words)))
    return res


# ════════════════════════════════════════════════════════════════
# BROWSER COMPONENTS (two-way: results come back to Python)
# ════════════════════════════════════════════════════════════════
VOICE_HTML = r"""<!doctype html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<link href="https://fonts.googleapis.com/css2?family=Manrope:wght@500;700;800&display=swap" rel="stylesheet">
<style>
:root{--accent:#FF6A3D;--text:#F4F1EA;--muted:#9C99A3;--line:rgba(255,255,255,.14)}
*{box-sizing:border-box}
html,body{margin:0;padding:0;background:transparent;color:var(--text);font-family:Manrope,system-ui,-apple-system,"Segoe UI",Roboto,sans-serif}
.card{display:flex;align-items:center;gap:16px;padding:16px;border:1px solid var(--line);border-radius:22px;background:linear-gradient(160deg,rgba(255,255,255,.07),rgba(255,255,255,.02))}
.mic{flex:none;position:relative;width:66px;height:66px;border-radius:50%;border:0;cursor:pointer;background:var(--accent);color:#16110F;display:grid;place-items:center;transition:transform .3s cubic-bezier(.2,.7,.2,1)}
.mic:hover{transform:scale(1.07)}.mic:active{transform:scale(.95)}
.mic svg{width:27px;height:27px}
.mic::after{content:"";position:absolute;inset:0;border-radius:50%;border:2px solid var(--accent);opacity:0}
.on .mic::after{animation:ring 1.5s ease-out infinite}
@keyframes ring{0%{transform:scale(1);opacity:.8}100%{transform:scale(1.75);opacity:0}}
.txt{min-width:0;flex:1}
.lbl{font-size:.68rem;letter-spacing:.15em;text-transform:uppercase;color:var(--muted);font-weight:800}
.heard{font-size:1.05rem;font-weight:700;margin-top:4px;overflow-wrap:anywhere;line-height:1.3}
.msg{font-size:.8rem;color:var(--muted);margin-top:4px;line-height:1.45}.msg.err{color:#ff8a7a}
.eq{display:none;gap:3px;height:18px;align-items:center;margin-top:8px}.on .eq{display:flex}
.eq i{width:3px;height:100%;background:var(--accent);border-radius:2px;transform:scaleY(.25);animation:eq .9s ease-in-out infinite}
.eq i:nth-child(2){animation-delay:.12s}.eq i:nth-child(3){animation-delay:.24s}.eq i:nth-child(4){animation-delay:.36s}.eq i:nth-child(5){animation-delay:.48s}
@keyframes eq{50%{transform:scaleY(1)}}
@media(prefers-reduced-motion:reduce){*{animation:none!important}}
</style></head><body>
<div class="card" id="card">
  <button class="mic" id="mic" aria-label="Start voice search">
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="9" y="2" width="6" height="12" rx="3"/><path d="M5 11a7 7 0 0 0 14 0M12 18v4"/></svg>
  </button>
  <div class="txt">
    <div class="lbl">Voice search</div>
    <div class="heard" id="heard">Tap the mic and speak</div>
    <div class="msg" id="msg">Try “cozy fantasy with dragons” or “books by Agatha Christie”</div>
    <div class="eq"><i></i><i></i><i></i><i></i><i></i></div>
  </div>
</div>
<script>
let lang="en-US",recog=null,listening=false,errored=false;
const $=id=>document.getElementById(id);
function post(type,data){window.parent.postMessage(Object.assign({isStreamlitMessage:true,type:type},data||{}),"*")}
function setHeight(){post("streamlit:setFrameHeight",{height:document.documentElement.scrollHeight})}
window.addEventListener("message",function(e){const d=e.data;if(d&&d.type==="streamlit:render"){lang=(d.args&&d.args.lang)||"en-US";setHeight()}});
post("streamlit:componentReady",{apiVersion:1});
new ResizeObserver(setHeight).observe(document.body);setHeight();
function msg(t,err){const m=$("msg");m.textContent=t;m.className="msg"+(err?" err":"")}
function setOn(v){listening=v;$("card").classList.toggle("on",v)}
function errText(c){return({"not-allowed":"Microphone blocked. Allow mic access for this site in your browser settings.","service-not-allowed":"Speech service is blocked in this browser.","no-speech":"I didn't hear anything. Tap the mic and try again.","audio-capture":"No microphone found.","network":"Network error while recognising speech."})[c]||("Voice error: "+c)}
$("mic").addEventListener("click",function(){
  const SR=window.SpeechRecognition||window.webkitSpeechRecognition;
  if(!SR){msg("Voice input isn't supported in this browser. Use Chrome, Edge or Safari, or type below.",true);return}
  if(listening&&recog){recog.stop();return}
  recog=new SR();recog.lang=lang;recog.interimResults=true;recog.continuous=false;recog.maxAlternatives=1;
  let finalText="";errored=false;
  recog.onstart=function(){setOn(true);$("heard").textContent="Listening…";msg("Say a title, an author or a vibe")};
  recog.onresult=function(ev){let interim="";for(let i=ev.resultIndex;i<ev.results.length;i++){const r=ev.results[i];if(r.isFinal)finalText+=r[0].transcript;else interim+=r[0].transcript}
    $("heard").textContent=(finalText+interim).trim()||"…"};
  recog.onerror=function(ev){errored=true;setOn(false);$("heard").textContent="Tap the mic and speak";msg(errText(ev.error),true)};
  recog.onend=function(){setOn(false);const t=finalText.trim();
    if(t){$("heard").textContent="“"+t+"”";msg("Searching…");post("streamlit:setComponentValue",{value:{text:t,ts:Date.now()},dataType:"json"})}
    else if(!errored){$("heard").textContent="Tap the mic and speak";msg("Didn't catch that. Tap and try again.",true)}};
  try{recog.start()}catch(e){msg("Could not start the microphone: "+e.message,true)}
});
</script></body></html>"""

FACE_HTML = r"""<!doctype html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<link href="https://fonts.googleapis.com/css2?family=Manrope:wght@500;700;800&display=swap" rel="stylesheet">
<style>
:root{--accent:#FF6A3D;--text:#F4F1EA;--muted:#9C99A3;--line:rgba(255,255,255,.14)}
*{box-sizing:border-box}
html,body{margin:0;padding:0;background:transparent;color:var(--text);font-family:Manrope,system-ui,-apple-system,"Segoe UI",Roboto,sans-serif}
.card{border:1px solid var(--line);border-radius:22px;padding:14px;background:linear-gradient(160deg,rgba(255,255,255,.07),rgba(255,255,255,.02))}
.stage{position:relative;width:100%;aspect-ratio:4/3;border-radius:16px;overflow:hidden;background:#0d0d12;display:grid;place-items:center}
video{position:absolute;inset:0;width:100%;height:100%;object-fit:cover;transform:scaleX(-1);opacity:0;transition:opacity .5s}
.on video{opacity:1}
.idle{text-align:center;color:var(--muted);font-size:.88rem;padding:1rem;line-height:1.5}
.idle b{display:block;font-size:2.6rem;margin-bottom:.3rem;animation:bob 3s ease-in-out infinite}
@keyframes bob{50%{transform:translateY(-6px)}}
.on .idle{display:none}
.scan{position:absolute;inset:14%;border:2px solid var(--accent);border-radius:30px;opacity:0;pointer-events:none}
.on .scan{opacity:1;animation:scan 1.5s ease-in-out infinite alternate}
@keyframes scan{from{transform:scale(.96);opacity:.45}to{transform:scale(1.03);opacity:1}}
.prog{height:4px;background:rgba(255,255,255,.09);border-radius:4px;margin-top:12px;overflow:hidden}
.prog i{display:block;height:100%;background:var(--accent);transform-origin:left;transform:scaleX(0)}
button.go{width:100%;margin-top:12px;padding:.9rem 1rem;border:0;border-radius:999px;background:var(--accent);color:#16110F;font-weight:800;font-size:.95rem;cursor:pointer;font-family:inherit;transition:transform .3s cubic-bezier(.2,.7,.2,1)}
button.go:hover{transform:translateY(-2px)}button.go:disabled{opacity:.6;cursor:wait;transform:none}
.status{margin-top:10px;font-size:.84rem;color:var(--muted);text-align:center;min-height:1.3em;line-height:1.4}
.status.err{color:#ff8a7a}.status.ok{color:#8ff0b0}
@media(prefers-reduced-motion:reduce){*{animation:none!important}}
</style></head><body>
<div class="card" id="card">
  <div class="stage" id="stage">
    <video id="v" playsinline muted autoplay></video>
    <div class="scan"></div>
    <div class="idle"><b>🙂</b>Your camera preview appears here.<br>Nothing is recorded or uploaded.</div>
  </div>
  <div class="prog"><i id="bar"></i></div>
  <button class="go" id="go">🎭 Detect my mood</button>
  <div class="status" id="status">Ready when you are.</div>
</div>
<script>
const LIB="https://cdn.jsdelivr.net/npm/@vladmandic/face-api@1.7.12/dist/face-api.js";
const MODELS="https://cdn.jsdelivr.net/npm/@vladmandic/face-api@1.7.12/model/";
const $=id=>document.getElementById(id);
let ready=false,stream=null;
function post(type,data){window.parent.postMessage(Object.assign({isStreamlitMessage:true,type:type},data||{}),"*")}
function setHeight(){post("streamlit:setFrameHeight",{height:document.documentElement.scrollHeight})}
window.addEventListener("message",function(){setHeight()});
post("streamlit:componentReady",{apiVersion:1});
new ResizeObserver(setHeight).observe(document.body);setHeight();
const sleep=ms=>new Promise(r=>setTimeout(r,ms));
function status(t,c){const s=$("status");s.textContent=t;s.className="status"+(c?" "+c:"")}
function loadScript(src){return new Promise(function(res,rej){if(window.faceapi)return res();const s=document.createElement("script");s.src=src;s.onload=res;s.onerror=function(){rej(new Error("Could not load the face model library. Check your connection."))};document.head.appendChild(s)})}
async function init(){if(ready)return;status("Loading the AI model (first time only)…");await loadScript(LIB);
  await faceapi.nets.tinyFaceDetector.loadFromUri(MODELS);await faceapi.nets.faceExpressionNet.loadFromUri(MODELS);ready=true}
function stop(){if(stream){stream.getTracks().forEach(t=>t.stop());stream=null}$("card").classList.remove("on");$("v").srcObject=null}
function friendly(e){if(e&&e.name==="NotAllowedError")return"Camera permission was denied. Allow camera access for this site and try again.";
  if(e&&e.name==="NotFoundError")return"No camera found on this device.";if(e&&e.name==="NotReadableError")return"The camera is being used by another app.";
  if(!window.isSecureContext)return"Camera needs a secure (https) connection.";return(e&&e.message)||"Something went wrong."}
$("go").addEventListener("click",async function(){
  const btn=$("go"),bar=$("bar");btn.disabled=true;bar.style.transform="scaleX(0)";
  try{
    await init();
    status("Requesting camera…");
    stream=await navigator.mediaDevices.getUserMedia({video:{facingMode:"user",width:{ideal:640},height:{ideal:480}},audio:false});
    const v=$("v");v.srcObject=stream;await v.play();
    $("card").classList.add("on");status("Look at the camera and relax your face…");
    await sleep(900);
    const opts=new faceapi.TinyFaceDetectorOptions({inputSize:224,scoreThreshold:0.35});
    const acc={};let hits=0;const t0=performance.now(),DUR=3500;
    while(performance.now()-t0<DUR){
      const d=await faceapi.detectSingleFace(v,opts).withFaceExpressions();
      if(d){hits++;for(const k in d.expressions){acc[k]=(acc[k]||0)+d.expressions[k]}}
      bar.style.transform="scaleX("+Math.min(1,(performance.now()-t0)/DUR)+")";
      await sleep(120);
    }
    stop();
    if(hits<3){status("No clear face detected. Try better lighting, face the camera and retry.","err");btn.disabled=false;return}
    const scores={};for(const k in acc){scores[k]=acc[k]/hits}
    const top=Object.keys(scores).sort((a,b)=>scores[b]-scores[a])[0];
    status("Mood captured ✓ Scroll down for your books.","ok");
    post("streamlit:setComponentValue",{value:{emotion:top,confidence:scores[top],scores:scores,ts:Date.now()},dataType:"json"});
    btn.textContent="🔁 Detect again";btn.disabled=false;
  }catch(err){stop();status(friendly(err),"err");btn.disabled=false}
});
window.addEventListener("pagehide",stop);
</script></body></html>"""

_COMP_ROOT = Path(tempfile.gettempdir()) / "moodreads_components_v2"


def _declare(name, html_src):
    d = _COMP_ROOT / name
    d.mkdir(parents=True, exist_ok=True)
    f = d / "index.html"
    if not f.exists() or f.read_text(encoding="utf-8") != html_src:
        f.write_text(html_src, encoding="utf-8")
    return components.declare_component(name, path=str(d))


voice_component = _declare("moodreads_voice", VOICE_HTML)
face_component = _declare("moodreads_face", FACE_HTML)


# ════════════════════════════════════════════════════════════════
# RENDERING
# ════════════════════════════════════════════════════════════════
def buy_links(b):
    q = urllib.parse.quote_plus(f"{b['title']} {b['author']}")
    return [("Amazon", f"https://www.{AMAZON_DOMAIN}/s?k={q}&i=stripbooks"),
            ("Bookshop", f"https://bookshop.org/search?keywords={q}"),
            ("Goodreads", f"https://www.goodreads.com/search?q={q}"),
            ("Google Books", f"https://www.google.com/search?tbm=bks&q={q}")]


def book_card_html(b, i=0):
    h = hue_of(b["title"])
    ph = f'<div class="ph" style="--h:{h};--h2:{(h + 40) % 360}"><span>{esc(b["title"])}</span></div>'
    img = (f'<img src="{esc(b["cover"])}" alt="" loading="lazy" referrerpolicy="no-referrer">'
           if b.get("cover") else "")
    match = f'<span class="match">{int(b["match"])}% match</span>' if b.get("match") else ""
    mood = b.get("mood") or (b["moods"][0] if b.get("moods") else "Curious")
    rating = ""
    if b.get("rating"):
        n = b.get("ratings_count") or 0
        rating = f'<div class="bk-r">★ {b["rating"]:.1f} <i>· {n:,} ratings</i></div>' if n else f'<div class="bk-r">★ {b["rating"]:.1f}</div>'
    why = f'<div class="bk-w">✦ Shared themes: {esc(b["why"])}</div>' if b.get("why") else ""
    pick = '<span class="tag">Editors’ pick</span>' if b.get("src") == "curated" else ""
    links = "".join(f'<a href="{esc(u)}" target="_blank" rel="noopener noreferrer">{esc(n)}</a>' for n, u in buy_links(b))
    return squash(f"""
    <article class="bk" style="--i:{i % 9}">
      <div class="bk-stage">{match}<div class="bk-cover">{ph}{img}</div></div>
      <div class="bk-tags"><span class="tag">{esc(b['genre'])}</span><span class="tag m">{esc(mood)}</span>{pick}</div>
      <div class="bk-t" role="heading" aria-level="3">{esc(b['title'])}</div>
      <div class="bk-a">{esc(b['author'])} · {esc(b['year'])}</div>
      {rating}
      <div class="bk-d">{esc(b['description'])}</div>
      {why}
      <div class="buy">{links}</div>
    </article>""")


def toggle_save(book):
    k = book_key(book)
    if k in st.session_state.saved:
        del st.session_state.saved[k]
    else:
        st.session_state.saved[k] = book


def clear_saved():
    st.session_state.saved = {}


def empty_state(title, text):
    st.markdown(f'<div class="empty"><b>{esc(title)}</b>{esc(text)}</div>', unsafe_allow_html=True)


def render_books(books, prefix, cols=3):
    if not books:
        empty_state("Nothing here yet", "Try another search, mood or genre.")
        return
    for r in range(0, len(books), cols):
        row = st.columns(cols, gap="medium")
        for j, (col, b) in enumerate(zip(row, books[r:r + cols])):
            with col:
                st.markdown(book_card_html(b, r + j), unsafe_allow_html=True)
                saved = book_key(b) in st.session_state.saved
                st.button("♥ Saved" if saved else "♡ Save to list",
                          key=f"sv_{prefix}_{r + j}_{zlib.crc32(book_key(b).encode())}",
                          on_click=toggle_save, args=(b,), type="primary" if saved else "secondary")


def section(kicker, title, sub=""):
    s = f'<div class="s">{esc(sub)}</div>' if sub else ""
    st.markdown(f'<div class="sec"><div class="k">{esc(kicker)}</div><div class="t" role="heading" aria-level="2">{esc(title)}</div>{s}</div>',
                unsafe_allow_html=True)


def hero_html():
    ring_titles = ["Dune", "Project Hail Mary", "The Midnight Library", "The Seven Husbands of Evelyn Hugo", "The Hobbit", "Atomic Habits"]
    items = ""
    for k, t in enumerate(ring_titles):
        b = BY_TITLE[t]
        h = hue_of(t)
        items += (f'<div class="it" style="--k:{k}"><div class="ph" style="--h:{h};--h2:{(h + 40) % 360}"><span>{esc(t)}</span></div>'
                  f'<img src="{esc(b["cover"])}" alt="" loading="eager" referrerpolicy="no-referrer"></div>')
    return squash(f"""
    <section class="hero">
      <div>
        <span class="eyebrow"><b></b>Mood · Voice · Face · AI</span>
        <div class="display" role="heading" aria-level="1">Find the book that fits <em>how you feel</em>.</div>
        <div class="lead">Skip the endless scroll. Tell MoodReads your mood — with a tap, your voice or even your expression — and get a shelf of books that match, with ways to buy each one.</div>
        <div class="stats">
          <div class="stat"><b>{len(CATALOG)}+</b><span>Curated reads</span></div>
          <div class="stat"><b>14</b><span>Moods</span></div>
          <div class="stat"><b>3</b><span>Input modes</span></div>
          <div class="stat"><b>4</b><span>Retailers</span></div>
        </div>
      </div>
      <div class="scene" aria-hidden="true"><div class="ring">{items}</div></div>
    </section>""")


def face_result_html(face, det_mood, mood):
    scores = sorted((face.get("scores") or {}).items(), key=lambda kv: -float(kv[1]))[:4]
    bars = "".join(
        f'<div class="bar"><span>{esc(EMOTION_LABEL.get(k, k.title()))}</span><div class="trk"><i style="width:{float(v) * 100:.0f}%"></i></div><em>{float(v) * 100:.0f}%</em></div>'
        for k, v in scores)
    emo = EMOTION_LABEL.get(face.get("emotion", ""), "—")
    note = f"Detected expression: {emo}" if mood == det_mood else f"Detected {emo} → {det_mood}, overridden by you"
    return squash(f"""
    <div class="fm"><div><div class="big">{esc(mood)}</div><div class="sub">{esc(note)}</div></div><div>{bars}</div></div>""")


# ════════════════════════════════════════════════════════════════
# STATE
# ════════════════════════════════════════════════════════════════
for _k, _v in {"saved": {}, "search": None, "last_voice_ts": None, "last_face_ts": None, "face": None}.items():
    if _k not in st.session_state:
        st.session_state[_k] = _v


def run_search(q):
    q = (q or "").strip()
    if not q:
        return
    with st.spinner(f"Searching for “{q}”…"):
        st.session_state.search = {"q": q, "books": search_books(q)}


# ════════════════════════════════════════════════════════════════
# APP
# ════════════════════════════════════════════════════════════════
tabs = st.tabs(["Home", "Face Mood", "Voice & Search", "Moods", "Genres", "AI Match", "Saved"])

# ── HOME ───────────────────────────────────────────────────────
with tabs[0]:
    st.markdown(hero_html(), unsafe_allow_html=True)

    section("Today’s shelf", "Six reads, picked fresh daily", "A hand-curated rotation. Every card links to four retailers.")
    picks = random.Random(dt.date.today().toordinal()).sample(CATALOG, 6)
    render_books(picks, "home_daily")

    section("Trending", "Popular with readers right now", "Live from Open Library's reading statistics.")
    with st.spinner("Loading live catalogue…"):
        popular = live_search('subject:"fiction"', 18, sort="readinglog", require_cover=True)
    popular = dedupe(popular, exclude={book_key(b) for b in picks})[:6]
    if popular:
        render_books(popular, "home_live")
    else:
        st.caption("The live catalogue is unreachable right now — the curated library works fully offline.")

    section("How it works", "Three ways in, one shelf out")
    st.markdown(squash("""
    <div class="steps">
      <div class="step"><div class="n">01</div><b>Tell us your mood</b><div>Tap a mood, speak a request, or let the camera read your expression in the browser.</div></div>
      <div class="step"><div class="n">02</div><b>We match the shelf</b><div>Moods map to genres, then curated picks and live Open Library data are merged and ranked. TF-IDF similarity powers “more like this”.</div></div>
      <div class="step"><div class="n">03</div><b>Read or buy</b><div>Save favourites to a reading list, export it, or jump to Amazon, Bookshop, Goodreads or Google Books.</div></div>
    </div>"""), unsafe_allow_html=True)

    section("Why it's different", "Mood-first, not purchase-history-first")
    st.markdown(squash("""
    <div class="tw"><table class="cmp">
      <tr><th>Feature</th><th>Typical store recommenders</th><th>MoodReads</th></tr>
      <tr><td>Starting point</td><td>Your past purchases &amp; ratings</td><td>How you feel right now</td></tr>
      <tr><td>Input</td><td>Clicks and text</td><td>Taps, voice, face expression, free text</td></tr>
      <tr><td>Algorithm</td><td>Collaborative filtering</td><td>Mood mapping + TF-IDF content similarity</td></tr>
      <tr><td>New visitors</td><td>Needs history first</td><td>Works from the first visit, no account</td></tr>
      <tr><td>Where to buy</td><td>One store</td><td>Four retailers</td></tr>
    </table></div>"""), unsafe_allow_html=True)

# ── FACE MOOD ──────────────────────────────────────────────────
with tabs[1]:
    section("Face mood", "Let your expression pick the book",
            "A 3-second scan reads your expression, maps it to a reading mood, and builds your shelf.")
    c1, c2 = st.columns(2, gap="large")
    with c1:
        face_val = face_component(key="face_cmp", default=None)
    with c2:
        st.markdown(squash("""
        <div class="panel"><b class="h">How it works</b>
          <ol><li>Press <b>Detect my mood</b> and allow the camera.</li><li>Look at the camera for about three seconds.</li><li>Your mood and matching books appear below.</li></ol>
          <b class="h" style="margin-top:.9rem">Privacy</b>
          Face analysis runs entirely in your browser. Video frames are never uploaded or stored. Only the model files are downloaded from a CDN.
          Camera access needs https (or localhost).</div>"""), unsafe_allow_html=True)

    if isinstance(face_val, dict) and face_val.get("ts") != st.session_state.last_face_ts:
        st.session_state.last_face_ts = face_val.get("ts")
        st.session_state.face = face_val
        st.session_state.face_override = None  # new scan resets any manual override

    face = st.session_state.face
    st.markdown("&nbsp;", unsafe_allow_html=True)
    override = st.pills("Camera not an option, or not quite right? Choose your mood:", ALL_MOODS,
                        selection_mode="single", key="face_override", format_func=mood_label)
    det_mood = EMOTION_TO_MOOD.get(face["emotion"], "Curious") if face else None
    mood = override or det_mood
    if face and mood:
        st.markdown(face_result_html(face, det_mood, mood), unsafe_allow_html=True)
    if mood:
        with st.spinner(f"Building your {mood.lower()} shelf…"):
            recs = recommend_for_mood(mood, 9)
        render_books(recs, "face")
    else:
        empty_state("No mood yet", "Run the scan above or pick a mood to see your shelf.")

# ── VOICE & SEARCH ─────────────────────────────────────────────
with tabs[2]:
    section("Voice & search", "Say it, or type it", "Search by title, author, topic or vibe. Curated picks and live Open Library results are combined.")
    langs = {"English (India)": "en-IN", "English (US)": "en-US", "English (UK)": "en-GB", "हिन्दी (Hindi)": "hi-IN"}
    lang_name = st.selectbox("Voice language", list(langs.keys()), key="voice_lang")
    voice_val = voice_component(lang=langs[lang_name], key="voice_cmp", default=None)
    if isinstance(voice_val, dict) and voice_val.get("ts") != st.session_state.last_voice_ts:
        st.session_state.last_voice_ts = voice_val.get("ts")
        run_search(voice_val.get("text", ""))

    with st.form("search_form", border=False):
        fc1, fc2 = st.columns([4, 1], vertical_alignment="bottom")
        with fc1:
            typed = st.text_input("Or type your search", placeholder="e.g. Harry Potter, Stephen King, space adventure")
        with fc2:
            go = st.form_submit_button("Search", type="primary")
    if go:
        if typed.strip():
            run_search(typed)
        else:
            st.warning("Type something to search, or use the mic.")
    st.caption("Voice uses your browser's speech service (Chrome, Edge, Safari). Audio may be processed by the browser vendor. MoodReads stores nothing.")

    s = st.session_state.search
    if s:
        st.markdown(f'<div class="sec" style="margin-top:1.4rem"><div class="k">{len(s["books"])} results</div><div class="t">“{esc(s["q"])}”</div></div>', unsafe_allow_html=True)
        if s["books"]:
            render_books(s["books"], "search")
        else:
            empty_state("No matches", "Try different keywords, or check your connection for live results.")
    else:
        empty_state("Your results appear here", "Tap the mic or type a search above.")

# ── MOODS ──────────────────────────────────────────────────────
with tabs[3]:
    section("Mood", "How are you feeling?", "Pick a mood; we map it to genres and merge curated and live picks.")
    mood_sel = st.pills("Mood", ALL_MOODS, selection_mode="single", key="mood_pick",
                        format_func=mood_label, label_visibility="collapsed")
    if mood_sel:
        gl = [g for g, ms in MOOD_MAP.items() if mood_sel in ms]
        st.markdown(f'<div class="sec" style="margin-top:1.4rem"><div class="k">Feeling {esc(mood_sel.lower())}</div>'
                    f'<div class="t">Genres to try: {esc(", ".join(gl[:4]))}</div></div>', unsafe_allow_html=True)
        with st.spinner("Finding your books…"):
            recs = recommend_for_mood(mood_sel, 12)
        render_books(recs, "mood")
    else:
        empty_state("Pick a mood", "Choose one above and your shelf appears instantly.")

# ── GENRES ─────────────────────────────────────────────────────
with tabs[4]:
    section("Genres", "Browse the shelves")
    genre_sel = st.pills("Genre", ["All"] + GENRES, selection_mode="single", default="All",
                         key="genre_pick", label_visibility="collapsed")
    order = st.selectbox("Order", ["Editors’ picks first", "Newest first", "Oldest first (classics)", "Title A–Z"], key="genre_order")
    with st.spinner("Loading shelf…"):
        gb = genre_books(genre_sel or "All", order)
    render_books(gb, "genre")

# ── AI MATCH ───────────────────────────────────────────────────
with tabs[5]:
    section("AI match", "Books that read like the ones you love",
            "TF-IDF turns every book into a vector of weighted words; cosine similarity finds the closest neighbours. Each card shows the shared themes behind the match.")
    mode = st.radio("Mode", ["More like a book I loved", "Describe my next read"], horizontal=True, label_visibility="collapsed", key="ai_mode")
    vec, mat = get_ai_model()
    if mode == "More like a book I loved":
        titles = sorted(BY_TITLE.keys())
        pick = st.selectbox("Pick a book you loved", titles, index=titles.index("Project Hail Mary"), key="ai_book")
        idx = [b["title"] for b in CATALOG].index(pick)
        res = ai_rank(mat[idx], exclude=idx, n=6)
        render_books(res, "ai_sim")
    else:
        with st.form("ai_form", border=False):
            desc = st.text_area("Describe it", height=100, placeholder="e.g. a hopeful story about second chances with a bit of science, or a dark mystery set in a snowy town")
            ai_go = st.form_submit_button("Find matches", type="primary")
        if ai_go:
            if desc.strip():
                st.session_state.ai_desc = {"q": desc.strip(), "res": ai_rank(vec.transform([desc]), n=6)}
            else:
                st.warning("Describe what you're in the mood for first.")
        d = st.session_state.get("ai_desc")
        if d:
            if d["res"]:
                render_books(d["res"], "ai_desc")
            else:
                empty_state("No close matches", "Try different words: genres, themes, feelings, settings.")
        else:
            empty_state("Describe your next read", "Write a sentence or two and the model finds the nearest books.")
    st.caption("Match % is relative to the best result in the list. Matching runs on the curated library.")

# ── SAVED ──────────────────────────────────────────────────────
with tabs[6]:
    saved = list(st.session_state.saved.values())
    section("Reading list", f"{len(saved)} saved" if saved else "Your reading list", "Saved for this session. Download it before you leave.")
    if saved:
        df = pd.DataFrame([{"Title": b["title"], "Author": b["author"], "Genre": b["genre"], "Year": b["year"],
                            "Amazon": buy_links(b)[0][1], "Goodreads": buy_links(b)[2][1]} for b in saved])
        d1, d2 = st.columns(2)
        with d1:
            st.download_button("⬇ Download as CSV", df.to_csv(index=False).encode("utf-8"), "moodreads_reading_list.csv", "text/csv", key="dl_csv")
        with d2:
            st.button("Clear list", on_click=clear_saved, key="clear_saved")
        render_books(saved, "saved")
    else:
        empty_state("Nothing saved yet", "Tap “Save to list” on any book and it lands here.")

st.markdown(squash(f"""
<div class="foot"><span>MoodReads · Book data from Open Library · Covers via Open Library Covers</span>
<span>Face analysis runs in your browser. No accounts, nothing stored.</span></div>"""), unsafe_allow_html=True)
