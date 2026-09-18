import os
import requests
from bs4 import BeautifulSoup
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# --------------------------------------------------
# 1. SCRAPE MOVIE DATA FROM THE WEB
# --------------------------------------------------

def scrape_movies():
    url = "https://en.wikipedia.org/wiki/List_of_horror_films_of_2025"

    request_headers = {
        "User-Agent": "MovieRecommendationProject/1.0"
    }

    try:
        response = requests.get(
            url,
            headers=request_headers,
            timeout=10
        )

        response.raise_for_status()

    except requests.RequestException as error:
        print("\nUnable to retrieve movie data from the web.")
        print("Please check your internet connection.")
        print(f"Error: {error}")
        return pd.DataFrame()

    soup = BeautifulSoup(response.text, "html.parser")

    # Find all tables on the webpage
    tables = soup.find_all("table")

    movie_table = None

    # Find the table containing movie information
    for table in tables:

        table_headers = [
            header.get_text(" ", strip=True).lower()
            for header in table.find_all("th")
        ]

        if (
            "title" in table_headers
            and "director" in table_headers
            and "cast" in table_headers
        ):
            movie_table = table
            break

    if movie_table is None:
        print("\nMovie table not found.")
        return pd.DataFrame()

    # Extract all rows from the movie table
    rows = movie_table.find_all("tr")

    movies = []

    for row in rows[1:]:

        cells = row.find_all("td")

        if len(cells) >= 5:

            title = cells[0].get_text(" ", strip=True)
            director = cells[1].get_text(" ", strip=True)
            cast = cells[2].get_text(" ", strip=True)
            country = cells[3].get_text(" ", strip=True)
            subgenre = cells[4].get_text(" ", strip=True)

            movies.append({
                "title": title,
                "director": director,
                "cast": cast,
                "country": country,
                "subgenre": subgenre
            })

    # Convert list into DataFrame
    df = pd.DataFrame(movies)

    # Create data folder if it doesn't exist
    os.makedirs("data", exist_ok=True)

    # Save extracted data
    df.to_csv("data/movies.csv", index=False)

    print(f"\nDataset created with {len(df)} movies.")

    return df


# --------------------------------------------------
# 2. LOAD DATASET
# --------------------------------------------------

def load_movies():

    file_path = "data/movies.csv"

    # If CSV already exists, use it
    if os.path.exists(file_path):

        print("Loading existing movie dataset...")

        return pd.read_csv(file_path)

    # Otherwise scrape the data
    print("Movie dataset not found.")
    print("Extracting movie data from the web...")

    return scrape_movies()


# --------------------------------------------------
# 3. LOAD MOVIE DATA
# --------------------------------------------------

df = load_movies()


# Check whether dataset was loaded successfully
if df.empty:

    print("\nNo movie data available.")
    print("Program stopped.")

    raise SystemExit


# Remove any accidental duplicate rows
df = df.drop_duplicates(subset="title").reset_index(drop=True)


# Make sure text columns do not contain missing values
df["title"] = df["title"].fillna("").astype(str)
df["director"] = df["director"].fillna("").astype(str)
df["cast"] = df["cast"].fillna("").astype(str)
df["country"] = df["country"].fillna("").astype(str)
df["subgenre"] = df["subgenre"].fillna("").astype(str)


# --------------------------------------------------
# 4. CREATE FEATURES
# --------------------------------------------------

df["features"] = (
    df["subgenre"] + " " +
    df["director"] + " " +
    df["cast"] + " " +
    df["country"]
)


# --------------------------------------------------
# 5. TF-IDF
# --------------------------------------------------

tfidf = TfidfVectorizer(
    stop_words="english"
)

tfidf_matrix = tfidf.fit_transform(
    df["features"]
)


# --------------------------------------------------
# 6. COSINE SIMILARITY
# --------------------------------------------------

cosine_sim = cosine_similarity(
    tfidf_matrix
)


# --------------------------------------------------
# 7. MOVIE RECOMMENDATION FUNCTION
# --------------------------------------------------

def recommend_movies(movie_title, top_n=5):

    # Remove extra spaces and convert input to lowercase
    search_title = movie_title.strip().lower()

    if not search_title:

        print("\nPlease enter a movie title.")
        return

    # --------------------------------------------------
    # First: Try exact title match
    # --------------------------------------------------

    matching_movies = df[
        df["title"].str.lower() == search_title
    ]

    # --------------------------------------------------
    # Second: Try partial title match
    # --------------------------------------------------

    if matching_movies.empty:

        matching_movies = df[
            df["title"]
            .str.lower()
            .str.contains(search_title, na=False)
        ]

    # --------------------------------------------------
    # If no movie was found
    # --------------------------------------------------

    if matching_movies.empty:

        print("\nMovie not found.")
        print("Please try another movie title.")

        return

    # --------------------------------------------------
    # If multiple movies were found
    # --------------------------------------------------

    if len(matching_movies) > 1:

        print("\nMultiple movies found:")

        for number, title in enumerate(
            matching_movies["title"],
            start=1
        ):

            print(f"{number}. {title}")

        choice = input(
            "Enter the number of the movie you want: "
        ).strip()

        # Check whether input is a number
        if not choice.isdigit():

            print("Invalid choice.")

            return

        choice = int(choice)

        # Check whether selected number is valid
        if choice < 1 or choice > len(matching_movies):

            print("Invalid choice.")

            return

        movie_index = matching_movies.index[choice - 1]

    else:

        movie_index = matching_movies.index[0]

    # --------------------------------------------------
    # Get similarity scores
    # --------------------------------------------------

    similarity_scores = list(
        enumerate(
            cosine_sim[movie_index]
        )
    )

    # Sort from highest similarity to lowest
    similarity_scores = sorted(
        similarity_scores,
        key=lambda x: x[1],
        reverse=True
    )

    # Remove the selected movie itself
    recommendations = similarity_scores[
        1:top_n + 1
    ]

    # --------------------------------------------------
    # Display recommendations
    # --------------------------------------------------

    print(
        f"\nMovies similar to: "
        f"{df.iloc[movie_index]['title']}"
    )

    print("-" * 50)

    for index, score in recommendations:

        print(
            f"{df.iloc[index]['title']} | "
            f"Similarity: {score:.3f}"
        )


# --------------------------------------------------
# 8. COMMAND LINE INTERFACE
# --------------------------------------------------

print("\n===== MOVIE RECOMMENDATION SYSTEM =====")
print("Type 'exit' to stop.")

while True:

    movie_title = input(
        "\nEnter movie title: "
    ).strip()

    # Exit the program
    if movie_title.lower() == "exit":

        print(
            "\nThank you for using the "
            "Movie Recommendation System!"
        )

        break

    # Generate Top-5 recommendations
    recommend_movies(
        movie_title,
        5
    )