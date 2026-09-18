# hybrid-recommendation-system
A modular movie recommendation system implementing content-based, collaborative filtering, and hybrid approaches.
# hybrid-recommendation-system

[🇺🇸 English](#-english-readme) | [🇮🇷 فارسی](#-%D8%B1%D8%A7%D9%87%D9%86%D9%85%D8%A7%DB%8C-%D9%81%D8%A7%D8%B1%D8%B3%DB%8C)

<a id="-english-readme"></a>

## 🇺🇸 English README

### 🎬 Overview

<code>hybrid-recommendation-system</code> is a modular movie recommendation project built with Python. It combines three layers:

- a reusable data loading, validation, and preprocessing pipeline;
- a content-based recommender built from movie genres;
- a collaborative recommender based on SVD matrix factorization.

The hybrid layer coordinates the two recommenders and separates movies recommended by both models from movies suggested by only one of them. The project is intentionally organized as independent modules so each part can be tested, improved, or replaced without rewriting the entire system.

The current implementation also supports using one or multiple seed movies as the input for content-based and hybrid recommendations.

### ✨ What this project demonstrates

- Clean separation between data preparation, feature engineering, modeling, recommendation, and evaluation.
- Content-based recommendations from one-hot encoded genre features and cosine similarity.
- Collaborative recommendations from user-rating history and Surprise SVD.
- Hybrid recommendation organization based on overlap between two independent Top-N lists.
- Validation reports, model evaluation reports, recommendation tables, and visual outputs.
- A practical project structure that can later be extended with an API, Docker, or a user interface.

### 🧭 System architecture

~~~mermaid
flowchart LR
    A[Raw movie and rating data] --> B[Load and validate]
    B --> C[Preprocess datasets]
    C --> D[Content-based pipeline]
    C --> E[Collaborative pipeline]
    D --> F[Hybrid coordinator]
    E --> F
    F --> G[Recommendations]
    D --> H[Reports and visualizations]
    E --> H
    F --> H
~~~

### 🗂️ Dataset and data pipeline

The project uses a MovieLens dataset with two raw CSV files:

| File | Main columns | Purpose |
| --- | --- | --- |
| <code>movies.csv</code> | <code>movieId</code>, <code>title</code>, <code>genres</code> | Movie metadata and content features |
| <code>ratings.csv</code> | <code>userId</code>, <code>movieId</code>, <code>rating</code>, <code>timestamp</code> | Historical user–movie interactions |

The raw files are expected under <code>data/raw/</code>. Because the dataset is relatively large, the repository tracks these files with Git LFS.

The shared data pipeline is responsible for:

1. Loading the raw CSV files through one consistent interface.
2. Checking dimensions, data types, missing values, and duplicate rows.
3. Extracting the release year from movie titles.
4. Cleaning movie titles and splitting the pipe-separated genre field.
5. Removing <code>timestamp</code> from the rating frame for the current models, since the implemented recommenders do not use temporal information yet.

This shared foundation allows all recommendation approaches to work with the same prepared data.

### 🎯 Content-based recommendation

The content-based module recommends movies that are similar to one or more input movies.

Its workflow is:

1. Convert each movie's genre list into a numerical genre matrix using one-hot encoding.
2. Represent every movie as a genre vector.
3. Build a preference profile from the selected movie or the average vector of multiple selected movies.
4. Calculate cosine similarity between the profile and all movie vectors.
5. Remove the input movies, rank the candidates, and return the Top-N results.

This approach is easy to explain: a movie is recommended because its genre profile is close to the user's selected movies. The main implementation is split into feature engineering, similarity calculation, and recommendation layers.

### 👥 Collaborative recommendation

The collaborative module learns from rating behavior instead of movie metadata.

It uses the Surprise <code>SVD</code> algorithm, which applies matrix factorization to discover latent relationships between users and movies. The training pipeline:

1. Converts the ratings DataFrame into a Surprise dataset.
2. Splits the interactions into training and test sets.
3. Trains the SVD model with configurable latent factors, learning rate, regularization, and epochs.
4. Evaluates rating prediction with RMSE and MAE.
5. Predicts ratings for candidate movies and returns the highest-rated unseen movies for a user.

The trained model and its trainset can be persisted with Joblib so recommendations can be generated later without retraining.

### 🔀 Hybrid recommendation

The hybrid module is a coordination layer rather than a weighted score-fusion model. It runs the content-based and collaborative recommenders independently and then organizes their results.

~~~mermaid
flowchart TD
    A[Content-based Top-N] --> C{Is the movie in both lists?}
    B[Collaborative Top-N] --> C
    C -->|Yes| D[Special: shared recommendation]
    C -->|Only content-based| E[Content-based section]
    C -->|Only collaborative| F[Collaborative section]
~~~

The hybrid workflow:

- removes movies already seen by the selected user;
- places movies returned by both recommenders in the <code>special</code> section;
- keeps content-only and collaborative-only movies in their own sections;
- preserves each model's source score and rank;
- uses the sum of source ranks only to order the shared <code>special</code> section, not to combine incomparable scores;
- validates input data, output columns, duplicate movie IDs, metadata integrity, overlap, diversity, and score summaries.

The current hybrid pipeline accepts multiple seed movie titles, converts them to movie IDs, and passes them to the content-based profile builder.

### 🧱 Project structure

~~~text
hybrid-recommendation-system/
├── data/
│   ├── raw/                      # Raw movies and ratings CSV files
│   └── processed/                # Reserved for future processed data
├── docs/                         # Module-level documentation
├── outputs/
│   ├── dataset_analysis/         # Data quality and preprocessing reports
│   ├── content_based/            # Content-based runtime outputs
│   ├── collaborative/            # Collaborative runtime outputs and models
│   ├── hybrid/                   # Hybrid runtime outputs
│   └── showcase/                 # Curated reports, artifacts, and visuals
├── scripts/
│   ├── dataset/                  # Dataset analysis, validation, preprocessing
│   ├── content_based/            # Feature, similarity, and recommendation runs
│   ├── collaborative/            # SVD training and recommendation runs
│   └── hybrid/                   # Hybrid generation and evaluation scripts
├── src/
│   ├── config/                   # Centralized project paths
│   ├── data/                     # Loading, validation, preprocessing
│   ├── content_based/            # Genre features and cosine similarity
│   ├── collaborative/            # SVD model, evaluation, recommendation
│   └── hybrid/                   # Hybrid coordination and evaluation
├── requirements.txt
├── pyproject.toml
└── LICENSE
~~~

### 🚀 Installation

Run the commands from the repository root:

~~~bash
python -m venv .venv
~~~

Windows PowerShell:

~~~powershell
.venv\Scripts\Activate.ps1
~~~

macOS/Linux:

~~~bash
source .venv/bin/activate
~~~

Install the project dependencies:

~~~bash
python -m pip install -r requirements.txt
~~~

Install the project in editable mode so the `src`-layout modules can be imported correctly during development:

~~~bash
python -m pip install -e .
~~~

If the raw dataset is not fully available after cloning, make sure Git LFS is installed and pull the tracked data:

~~~bash
git lfs install
git lfs pull
~~~

### ▶️ Running the pipelines

All commands below should be executed from the repository root.

#### 1. Inspect and prepare the dataset

~~~bash
python scripts/dataset/analyze_dataset.py
python scripts/dataset/validate_dataset.py
python scripts/dataset/preprocess_dataset.py
~~~

#### 2. Run the content-based pipeline

~~~bash
python scripts/content_based/run_feature_engineering.py
python scripts/content_based/run_similarity.py
python scripts/content_based/run_recommender.py
~~~

For a small interactive-style check with selected movie titles:

~~~bash
python scripts/content_based/test_content_based.py
~~~

#### 3. Train and use the collaborative model

Train, evaluate, save the model, and generate a sample recommendation list:

~~~bash
python scripts/collaborative/run_svd.py
~~~

After the model artifacts have been created, the test script can load the model and ask for a user ID:

~~~bash
python scripts/collaborative/test_collaborative.py
~~~

#### 4. Run the hybrid pipeline

~~~bash
python scripts/hybrid/run_hybrid.py
~~~

The pipeline trains the collaborative component, builds the content-based genre representation, generates both recommendation lists, and writes separate hybrid sections for shared, content-only, and collaborative-only movies.

The current evaluation module can then be run with:

~~~bash
python src/hybrid/evaluate_hybrid.py
~~~

The user ID, seed movie titles, and requested Top-N size are configured in the corresponding scripts, so they can be adjusted without changing the core recommendation classes.

### 📦 Outputs and showcase

Runtime artifacts are organized under <code>outputs/</code> by pipeline. Depending on the module, the generated results include recommendation tables, evaluation reports, feature artifacts, serialized model data, and visualizations.

The <code>outputs/showcase/</code> directory contains curated examples from the dataset analysis, content-based, and collaborative parts of the project. Each area is kept separately so readers can inspect the result of that stage without mixing it with another model's output. Model-related artifacts and generated examples are also grouped by module there.

Hybrid runtime outputs are not included in the showcase yet. They are generated under <code>outputs/hybrid/</code> when the hybrid pipeline is executed and can be added to the showcase later when the hybrid presentation is finalized.

### 📚 Additional documentation

- [Dataset pipeline](docs/dataset.md)
- [Content-based recommendation](docs/content_based.md)
- [Collaborative filtering](docs/collaborative.md)
- [Hybrid recommendation](docs/hybrid.md)

### 🛠️ Technology stack

- Python
- Pandas and NumPy for data processing
- Scikit-learn for cosine similarity
- Scikit-surprise for SVD matrix factorization
- Joblib for saving and loading model artifacts
- Matplotlib and Seaborn support for visual outputs

### 🔮 Future direction

The project is structured so the next stages can be added without changing its core idea. Planned directions include:

- Docker-based setup and reproducible execution;
- API exposure for recommendation requests;
- a lightweight application or user interface;
- richer movie features and more advanced ranking strategies;
- improved experiment tracking and deployment-oriented evaluation.

### 📄 License

This project is released under the MIT License. See [LICENSE](LICENSE) for details.


# ⭐ Note

This project was developed primarily for **learning and understanding recommender system concepts**.

The main focus of this project is to explore the architecture, implementation, and evaluation of different recommendation approaches, as well as how they can be combined within a "hybrid" recommendation system.

If you find this project useful, I would greatly appreciate your support by giving it a ⭐.


[⬆️ Back to language selection](#hybrid-recommendation-system)

<a id="-راهنمای-فارسی"></a>

## 🇮🇷 راهنمای فارسی

### 🎬 معرفی پروژه

یک پروژه‌ی ماژولار برای پیشنهاد فیلم است که با Python ساخته شده است. این پروژه سه بخش اصلی را در کنار یکدیگر قرار می‌دهد:

- یک pipeline مشترک برای بارگذاری، اعتبارسنجی و آماده‌سازی داده‌ها؛
- یک سیستم پیشنهاددهنده‌ی content-based بر اساس ژانر فیلم‌ها؛
- یک سیستم پیشنهاددهنده‌ی collaborative بر پایه‌ی فاکتورگیری ماتریس با الگوریتم SVD.

بخش hybrid این دو پیشنهاددهنده را به‌صورت مستقل اجرا می‌کند و نتایج را به سه گروه تقسیم می‌کند: فیلم‌هایی که هر دو مدل پیشنهاد داده‌اند، فیلم‌هایی که فقط مدل content-based پیشنهاد داده است و فیلم‌هایی که فقط مدل collaborative پیشنهاد داده است. ساختار پروژه به‌صورت ماژولار طراحی شده است تا هر بخش را بتوان مستقل آزمایش، بهبود یا جایگزین کرد؛ بدون اینکه لازم باشد کل سیستم از ابتدا بازنویسی شود.

در پیاده‌سازی فعلی، برای پیشنهاددهی content-based و hybrid می‌توان یک فیلم یا چند فیلم را به‌عنوان فیلم‌های اولیه انتخاب کرد.

### ✨ این پروژه چه چیزهایی را نشان می‌دهد؟

- جداسازی تمیز و مشخصِ آماده‌سازی داده، مهندسی ویژگی، مدل‌سازی، پیشنهاددهی و ارزیابی؛
- پیشنهاد فیلم بر اساس ویژگی‌های ژانری و cosine similarity؛
- پیشنهاد فیلم بر اساس سابقه‌ی امتیازدهی کاربران و SVD؛
- سازمان‌دهی نتایج hybrid بر اساس اشتراک خروجی دو مدل مستقل؛
- تولید گزارش‌های اعتبارسنجی، گزارش‌های ارزیابی مدل، جدول‌های پیشنهاد و خروجی‌های تصویری؛
- ساختاری کاربردی و قابل توسعه که در مراحل بعدی می‌توان API، Docker یا رابط کاربری را به آن اضافه کرد.

### 🧭 معماری کلی سیستم

~~~mermaid
flowchart LR
    A[داده‌ی خام فیلم و امتیازها] --> B[بارگذاری و اعتبارسنجی]
    B --> C[پیش‌پردازش داده]
    C --> D[محتوایی Pipeline]
    C --> E[مشارکتی Pipeline]
    D --> F[Hybrid هماهنگ‌کننده‌ی]
    E --> F
    F --> G[پیشنهادها]
    D --> H[گزارش‌ها و نمودارها]
    E --> H
    F --> H
~~~

### 🗂️ داده و pipeline آماده‌سازی

پروژه از دیتاست MovieLens استفاده می‌کند و دارای دو فایل CSV اصلی است:

| فایل | ستون‌های اصلی | کاربرد |
| --- | --- | --- |
| `movies.csv` | `movieId`، `title`، `genres` | اطلاعات فیلم و ویژگی‌های محتوایی |
| `ratings.csv` | `userId`، `movieId`، `rating`، `timestamp` | سابقه‌ی تعامل کاربران با فیلم‌ها |

فایل‌های خام باید داخل `data/raw/` قرار داشته باشند. چون حجم داده نسبتاً زیاد است، این فایل‌ها در مخزن با Git LFS مدیریت می‌شوند.

:مشترک داده وظایف زیر را انجام می‌دهد pipeline

1. فایل‌های CSV خام را از یک مسیر و با یک رابط یکسان بارگذاری می‌کند.
2. تعداد سطر و ستون، نوع داده‌ها، مقادیر خالی و رکوردهای تکراری را بررسی می‌کند.
3. سال انتشار را از عنوان فیلم استخراج می‌کند.
4. عنوان فیلم را پاک‌سازی می‌کند و مقدار چندبخشیِ ژانرها را به یک فهرست تبدیل می‌کند.
5. ستون `timestamp` را برای مدل‌های فعلی از داده‌های امتیازدهی حذف می‌کند؛ زیرا پیشنهاددهنده‌های فعلی هنوز از اطلاعات زمانی استفاده نمی‌کنند.

به این ترتیب، هر سه رویکرد پیشنهاددهی از یک پایه‌ی داده‌ی مشترک و قابل اعتماد استفاده می‌کنند.

### 🎯 پیشنهاددهی مبتنی بر محتوا

ماژول content-based فیلم‌هایی را پیشنهاد می‌دهد که از نظر ویژگی‌های محتوایی به یک یا چند فیلم انتخاب‌شده شباهت دارند.

روند کار آن به‌صورت خلاصه به این شکل است:

1. فهرست ژانرهای هر فیلم را با one-hot encoding به یک ماتریس عددی تبدیل می‌کند.
2. هر فیلم را به شکل یک بردار ژانری نمایش می‌دهد.
3. برای یک فیلم انتخاب‌شده، یا میانگین بردار چند فیلم انتخاب‌شده، یک profile ایجاد می‌کند.
4. شباهت cosine را بین profile و بردار همه‌ی فیلم‌ها محاسبه می‌کند.
5. خودِ فیلم‌های ورودی را حذف می‌کند، نتایج را بر اساس میزان شباهت مرتب می‌کند و Top-N را برمی‌گرداند.

مزیت این روش این است که دلیل پیشنهاد قابل توضیح است: یک فیلم پیشنهاد می‌شود چون الگوی ژانری آن به فیلم‌های انتخاب‌شده نزدیک است. این بخش از سه لایه‌ی اصلی تشکیل شده است: مهندسی ویژگی، محاسبه‌ی شباهت و recommendation.

### 👥 پیشنهاددهی مشارکتی

ماژول collaborative به‌جای تکیه بر اطلاعات ژانری، از رفتار امتیازدهی کاربران یاد می‌گیرد.

در این بخش از الگوریتم `SVD` کتابخانه‌ی Surprise استفاده شده است. الگوریتم SVD با فاکتورگیری ماتریس، رابطه‌های پنهان میان کاربران و فیلم‌ها را یاد می‌گیرد. روند آموزش شامل مراحل زیر است:

1. تبدیل DataFrame امتیازها به دیتاست قابل استفاده برای Surprise؛
2. جدا کردن داده‌ها به دو بخش آموزش و آزمون؛
3. آموزش مدل SVD با پارامترهایی مانند تعداد فاکتورهای نهفته، نرخ یادگیری، regularization و تعداد epoch؛
4. ارزیابی پیش‌بینی امتیازها با RMSE و MAE؛
5. پیش‌بینی امتیاز فیلم‌های مناسب و برگرداندن فیلم‌های ندیده‌ی کاربر با بیشترین امتیاز تخمینی.

مدل آموزش‌دیده و trainset آن با Joblib ذخیره می‌شوند تا در اجراهای بعدی نیازی به آموزش دوباره‌ی مدل نباشد.

### 🔀 پیشنهاددهی Hybrid

بخش hybrid در این پروژه یک لایه‌ی هماهنگ‌کننده است، نه یک مدل weighted score fusion. ابتدا دو پیشنهاددهنده به‌صورت مستقل اجرا می‌شوند و سپس نتایج آن‌ها در کنار یکدیگر سازمان‌دهی می‌شوند.

~~~mermaid
flowchart TD
    A[محتوایی Top-N] --> C{فیلم در هر دو فهرست هست؟}
    B[مشارکتی Top-N] --> C
    C -->|بله| D[Special پیشنهاد مشترک]
    C -->|فقط محتوایی| E[Content-based بخش]
    C -->|فقط مشارکتی| F[Collaborative بخش]
~~~

روند hybrid دارای ویژگی‌های زیر است:

- فیلم‌هایی را که کاربر قبلاً دیده یا به آن‌ها امتیاز داده است حذف می‌کند؛
- فیلم‌های مشترک میان دو مدل را در بخش `special` قرار می‌دهد؛
- فیلم‌هایی را که فقط یکی از مدل‌ها پیشنهاد داده است، در بخش مربوط به همان مدل نگه می‌دارد؛
- امتیاز و رتبه‌ی تولیدشده توسط هر مدل را حفظ می‌کند؛
- مجموع رتبه‌ها را فقط برای مرتب‌سازی بخش مشترک استفاده می‌کند و امتیازهای دو مدل را مستقیماً با یکدیگر ترکیب نمی‌کند؛
- صحت داده‌های ورودی، ستون‌های خروجی، شناسه‌های تکراری، کامل بودن metadata، میزان اشتراک، تنوع و خلاصه‌ی امتیازها را بررسی می‌کند.

بخش hybrid چند عنوان فیلم را به‌عنوان ورودی اولیه دریافت می‌کند، شناسه‌ی هر فیلم را پیدا می‌کند و سپس این شناسه‌ها را برای ساخت profile محتوایی به ماژول content-based می‌دهد.

### 🧱 ساختار پروژه

~~~text
hybrid-recommendation-system/
├── data/
│   ├── raw/                      # فایل‌های خام فیلم‌ها و امتیازها
│   └── processed/                # مسیر آماده برای داده‌ی پردازش‌شده‌ی آینده
├── docs/                         # مستندات هر ماژول
├── outputs/
│   ├── dataset_analysis/         # گزارش‌های بررسی و آماده‌سازی داده
│   ├── content_based/            # content-based خروجی‌های اجرای 
│   ├── collaborative/            # collaborative خروجی‌ها و مدل‌های 
│   ├── hybrid/                   # hybrid خروجی‌های اجرای 
│   └── showcase/                 # و خروجی‌های نمایشی artifact گزارش‌ها
├── scripts/
│   ├── dataset/                  # تحلیل، اعتبارسنجی و پیش‌پردازش داده
│   ├── content_based/            # شباهت و پیشنهاددهی feature اجرای 
│   ├── collaborative/            # و پیشنهاددهی SVD آموزش 
│   └── hybrid/                   # hybrid تولید و ارزیابی خروجی 
├── src/
│   ├── config/                   # مسیرهای مرکزی پروژه
│   ├── data/                     # بارگذاری، اعتبارسنجی و پیش‌پردازش
│   ├── content_based/            # cosine similarity ویژگی‌های ژانری و 
│   ├── collaborative/            # SVD ارزیابی و پیشنهاد دهی مدل  
│   └── hybrid/                   # hybrid هماهنگی و ارزیابی 
├── requirements.txt
├── pyproject.toml
└── LICENSE
~~~

### 🚀 نصب و راه‌اندازی

دستورهای زیر را از ریشه‌ی مخزن اجرا کنید:

~~~bash
python -m venv .venv
~~~

در Windows PowerShell:

~~~powershell
.venv\Scripts\Activate.ps1
~~~

در macOS/Linux:

~~~bash
source .venv/bin/activate
~~~

سپس وابستگی‌های پروژه را نصب کنید:

~~~bash
python -m pip install -r requirements.txt
~~~

اگر بعد از clone فایل‌های داده به‌طور کامل دریافت نشدند، Git LFS را نصب کرده و داده‌ها را دریافت کنید:

~~~bash
git lfs install
git lfs pull
~~~

### ▶️ اجرای pipelineها

همه‌ی دستورهای زیر باید از ریشه‌ی پروژه اجرا شوند.

#### ۱. بررسی و آماده‌سازی داده

~~~bash
python scripts/dataset/analyze_dataset.py
python scripts/dataset/validate_dataset.py
python scripts/dataset/preprocess_dataset.py
~~~

#### ۲. اجرای pipeline محتوایی

~~~bash
python scripts/content_based/run_feature_engineering.py
python scripts/content_based/run_similarity.py
python scripts/content_based/run_recommender.py
~~~

برای یک بررسی ساده با چند عنوان فیلم انتخابی:

~~~bash
python scripts/content_based/test_content_based.py
~~~

#### ۳. آموزش و استفاده از مدل مشارکتی

برای آموزش، ارزیابی، ذخیره‌ی مدل و تولید یک نمونه پیشنهاد:

~~~bash
python scripts/collaborative/run_svd.py
~~~

پس از ساخته شدن artifactهای مدل، اسکریپت آزمایشی مدل را بارگذاری می‌کند و شناسه‌ی کاربر را از شما می‌گیرد:

~~~bash
python scripts/collaborative/test_collaborative.py
~~~

#### ۴. اجرای pipeline Hybrid

~~~bash
python scripts/hybrid/run_hybrid.py
~~~

این pipeline بخش collaborative را آموزش می‌دهد، ماتریس ژانری را می‌سازد، هر دو فهرست پیشنهاد را تولید می‌کند و خروجی را به سه بخش تقسیم می‌کند: پیشنهادهای مشترک، پیشنهادهای فقط محتوایی و پیشنهادهای فقط مشارکتی.

برای ارزیابی خروجی تولیدشده، می‌توانید ماژول ارزیابی فعلی را اجرا کنید:

~~~bash
python src/hybrid/evaluate_hybrid.py
~~~

شناسه‌ی کاربر، عنوان فیلم‌های اولیه و تعداد پیشنهادها در اسکریپت‌های مربوطه تنظیم شده‌اند و بدون نیاز به تغییر در کلاس‌های اصلی قابل ویرایش هستند.

### 📦 خروجی‌ها و Showcase

خروجی‌های زمان اجرا داخل `outputs/` و بر اساس pipeline دسته‌بندی می‌شوند. بسته به ماژول، این خروجی‌ها می‌توانند شامل جدول پیشنهادها، گزارش ارزیابی، artifactهای feature، داده‌های ذخیره‌شده‌ی مدل و نمودارها باشند.

پوشه‌ی `outputs/showcase/` نمونه‌های مرتب‌شده و قابل نمایش از بخش تحلیل داده، content-based و collaborative را نگه می‌دارد. هر بخش به‌صورت جداگانه قرار گرفته است تا نتیجه‌ی هر مرحله با خروجی مدل دیگر مخلوط نشود. فایل‌های مربوط به مدل و نمونه‌های تولیدشده نیز در همین ساختار ماژولار دسته‌بندی شده‌اند.


### 📚 مستندات تکمیلی

- [مستندات pipeline داده](docs/dataset.md)
- [مستندات پیشنهاددهی محتوایی](docs/content_based.md)
- [مستندات فیلتر مشارکتی](docs/collaborative.md)
- [مستندات پیشنهاددهی Hybrid](docs/hybrid.md)

### 🛠️ فناوری‌های استفاده‌شده

* پایتون `Python`
* پانداس `Pandas` و نام‌پای `NumPy` برای پردازش داده؛
* سایکیت‌لرن `Scikit-learn` برای محاسبه‌ی شباهت کسینوسی؛
* سایکیت‌سورپرایز `Scikit-Surprise` برای فاکتورگیری ماتریس با روش `SVD`؛
* جاب‌لیب `Joblib` برای ذخیره و بارگذاری `Artifact`های مدل؛
* مت‌پلات‌لیب `Matplotlib` و سی‌بورن `Seaborn` برای تولید خروجی‌های تصویری.

### 🔮 مسیر توسعه‌ی آینده

ساختار پروژه به‌گونه‌ای طراحی شده است که قابلیت‌های بعدی بدون تغییر در ایده‌ی اصلی پروژه به آن اضافه شوند. مسیرهای توسعه‌ی آینده شامل موارد زیر هستند:

- آماده‌سازی اجرای reproducible با Docker؛
- ارائه‌ی سرویس پیشنهاددهی از طریق API؛
- ساخت یک رابط کاربری یا application سبک؛
- استفاده از ویژگی‌های غنی‌تر فیلم و روش‌های رتبه‌بندی پیشرفته‌تر؛
- بهبود ثبت آزمایش‌ها و ارزیابی مناسب برای deployment.

### 📄 مجوز

این پروژه تحت مجوز MIT منتشر شده است. برای مشاهده‌ی جزئیات، به فایل [LICENSE](LICENSE) مراجعه کنید.


# ⭐ نکته

این پروژه با هدف **یادگیری و درک مفاهیم سیستم‌های پیشنهاددهنده** توسعه داده شده است.

تمرکز اصلی پروژه بر درک معماری، پیاده‌سازی و ارزیابی روش‌های مختلف پیشنهاددهی و نحوه‌ی ترکیب آن‌ها در یک سیستم "hybrid" است.

اگر این پروژه برای شما مفید بود، خوشحال می‌شوم با دادن یک ⭐ از آن حمایت کنید.


[⬆️ بازگشت به انتخاب زبان](#hybrid-recommendation-system)
