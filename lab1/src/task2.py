import pandas as pd


def main():
    data = {
        "Rank": ["1", "2", "3-5", "3-5", "3-5", "6-7", "6-7", "8", "9", "10", "11", "12", "13", "14", "15", "16", "17",
                 "18",
                 "19", "20"],
        "University": [
            "УКУ", "ХНЕУ", "КНУ ім. Шевченка", "НаУКМА", "ОНУ ім. Мечникова",
            "ДУТ", "СумДУ", "НТУУ «КПІ ім. І. Сікорського»", "ЛНУ ім. Франка", "ХНУРЕ",
            "НУ «Львівська політехніка»", "ДНУ ім. Гончара", "НТУ «Дніпровська політехніка»",
            "ХНУ м. Хмельницький", "КНЕУ", "НУ «Одеська політехніка» (ОНПУ)",
            "ХАІ", "ЧНУ ім. Федьковича", "ХНУ ім. Каразіна", "НТУ «ХПІ»"
        ],
        "cities": ["Lviv", "Kharkiv", "Kyiv", "Kyiv", "Odesa", "Kyiv", "Sumy", "Kyiv", "Lviv", "Kharkiv", "Lviv",
                   "Dnipro", "Dnipro", "Khmelnytskyi", "Kyiv", "Odesa", "Kharkiv", "Chernivtsi", "Kharkiv", "Kharkiv"],
        "Developer Salary ($)": [3700, 2950, 2950, 2650, 2850, 2950, 3000, 3000, 2800, 2950, 2850, 3000, 2700, 2900,
                                 2800,
                                 2900, 2800, 2600, 2700, 2700],
        "English Level (%)": [83, 56, 73, 79, 70, 46, 43, 58, 63, 53, 58, 48, 51, 48, 59, 48, 41, 42, 59, 42],
        "Student Recommendation": [9.3, 8.3, 6.7, 8.7, 7.2, 6.4, 6.4, 7.1, 6.8, 6.9, 6.4, 7.4, 6.7, 6.3, 5.6, 6.4, 7.3,
                                   8.4, 5.5, 7.3],
        "Graduates-developers (%)": [1, 1, 6, 2, 1, 1, 1, 13, 4, 4, 9, 2, 1, 1, 2, 1, 1, 1, 2, 2]
    }

    df = pd.DataFrame(data)

    high_recommendation = df[df["Student Recommendation"] > 7.0][["University", "Rank"]]
    print("Universities with Student Recommendation higher than 7.0:")
    print(high_recommendation)

    high_english_level = df[df["English Level (%)"] > 50][["University", "English Level (%)"]]
    print("\nUniversities with English Level higher than 50%:")
    print(high_english_level)

    highest_graduates_developers = df[df["Graduates-developers (%)"] == df["Graduates-developers (%)"].max()][["University", "Graduates-developers (%)"]]
    print("\nUniversity with highest graduates-developers:")
    print(highest_graduates_developers)

    average_salary = df["Developer Salary ($)"].mean()
    below_average_salary = df[df["Developer Salary ($)"] < average_salary][["University", "Developer Salary ($)"]]
    print("\nUniversities with Developer Salary below average:")
    print(below_average_salary)

    universities_per_city = df["cities"].value_counts()
    print("\nNumber of universities per city:")
    print(universities_per_city)


if __name__ == "__main__":
    main()
