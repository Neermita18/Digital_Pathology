import sqlite3
import pandas as pd

def load_annotations(db_path):
    conn = sqlite3.connect(db_path)
    slides = pd.read_sql_query("SELECT * FROM Slides", conn)
    coords = pd.read_sql_query("SELECT * FROM Annotations_coordinates", conn)
    annos = pd.read_sql_query("SELECT * FROM Annotations", conn)
    classes = pd.read_sql_query("SELECT * FROM Classes", conn)
    conn.close()
    return slides, coords, annos, classes
