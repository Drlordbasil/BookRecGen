import requests


class BookScraper:
    def scrape_books(self):
        response = requests.get("https://openlibrary.org/subjects/fiction.json")
        data = response.json()

        books = []
        for work in data["works"]:
            title = work["title"]
            url = work["url"]
            if title and url:
                books.append((title, url))

        return books

    def scrape_book_details(self, book_url):
        response = requests.get(f"https://openlibrary.org{book_url}.json")
        data = response.json()

        book_details = {
            "title": data.get("title"),
            "author": data.get("authors")[0]["name"],
            "description": data.get("description"),
            "genre": data.get("subjects"),
        }

        return book_details


class BookRecommendationSystem:
    def __init__(self, books):
        self.books = books

    def gather_user_preferences(self):
        favorite_genres = ["Mystery", "Horror", "Romance"]
        favorite_authors = ["Stephen King", "J.K. Rowling"]
        return favorite_genres + favorite_authors

    def preprocess_text(self, text):
        return text.lower().strip()

    def calculate_similarities(self, user_preferences):
        book_titles = [book[0] for book in self.books]

        preprocessed_titles = [self.preprocess_text(title) for title in book_titles]
        preprocessed_preferences = [
            self.preprocess_text(pref) for pref in user_preferences
        ]

        vectorizer = TfidfVectorizer()
        title_matrix = vectorizer.fit_transform(preprocessed_titles)
        preference_matrix = vectorizer.transform(preprocessed_preferences)
        similarities = cosine_similarity(preference_matrix, title_matrix).flatten()

        return similarities

    def recommend_books(self, user_preferences, num_recommendations=5):
        similarities = self.calculate_similarities(user_preferences)

        top_indices = similarities.argsort()[-num_recommendations:][::-1]

        recommendations = []
        for idx in top_indices:
            book = self.books[idx]
            recommendations.append(book)

        return recommendations


if __name__ == "__main__":
    book_scraper = BookScraper()
    scraped_books = book_scraper.scrape_books()

    book_recommendation_system = BookRecommendationSystem(scraped_books)

    user_preferences = book_recommendation_system.gather_user_preferences()

    recommendations = book_recommendation_system.recommend_books(user_preferences)

    for idx, book in enumerate(recommendations, 1):
        book_title, book_url = book
        book_details = book_scraper.scrape_book_details(book_url)
        print(
            f"Recommendation #{idx}: {book_details['title']} by {book_details['author']}"
        )
        print(f"Genre: {book_details['genre']}")
        print(f"Description: {book_details['description']}")
        print(f"Book URL: {book_url}")
        print("=" * 50)
