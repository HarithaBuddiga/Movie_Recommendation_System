# Movie Recommendation System

## Project Overview

This project is a CLI-based Movie Recommendation System using
content-based filtering.

Movie data is extracted from the web and stored in a CSV dataset.
The system processes movie information such as subgenre, director,
cast, and country to recommend movies with similar characteristics.

## Features

- Web-based movie data extraction
- Automatic CSV dataset creation
- Content-based movie recommendation
- TF-IDF text feature extraction
- Cosine similarity calculation
- Top 5 movie recommendations
- Case-insensitive movie search
- Partial movie title search
- CLI-based interaction

## Technologies Used

- Python
- Requests
- BeautifulSoup
- Pandas
- Scikit-learn
- TF-IDF
- Cosine Similarity

## Dataset

The movie dataset is extracted from a publicly accessible Wikipedia
movie list using Python web scraping.

The current dataset contains 237 movies.

## How It Works

1. Movie information is extracted from the web.
2. The extracted information is stored in `movies.csv`.
3. Movie features are combined into a single text feature.
4. TF-IDF converts the text features into numerical vectors.
5. Cosine similarity measures similarity between movies.
6. The system returns the top 5 similar movies.
7. Users interact with the system through the command line.

## Project Structure

Movie_Recommendation_System/
│
├── data/
│   └── movies.csv
│
├── movie_recommender.py
├── requirements.txt
└── README.md

## How to Run

Install the required packages:

pip install -r requirements.txt

Then run:

python movie_recommender.py

## Example

Enter movie title: Sinners

Movies similar to: Sinners

Bitter Souls | Similarity: 0.127
28 Years Later | Similarity: 0.114
Anaconda | Similarity: 0.093

## Recommendation Method

This project uses content-based filtering.

TF-IDF is used for feature representation and cosine similarity
is used to calculate the similarity between movies.