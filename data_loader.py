import os
import re
import pandas as pd
from docx import Document


class DataLoader:
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.data = []

        self.questions = [
            'תמונה 1', 'תמונה 2', 'תמונה 3', 'תמונה 4', 'תמונה 5', 'תמונה 6',
            'תמונה 7', 'תמונה 8', 'תמונה 9', 'תמונה 10', 'תמונה 11', 'תמונה 12',
            'תמונה 13', 'תמונה 14', 'בר מצווה', 'מה אוהב לעשות', 'מה מעצבן',
            'משאלות לעתיד'
        ]

        self.question_mapping = {
            'תמונה 1': 'תמונה 1',
            'תמונה 2': 'תמונה 2',
            'תמונה 3': 'תמונה 3',
            'תמונה 4': 'תמונה 4',
            'תמונה 5': 'תמונה 5',
            'תמונה 6': 'תמונה 6',
            'תמונה 7': 'תמונה 7',
            'תמונה 8': 'תמונה 8',
            'תמונה 9': 'תמונה 9',
            'תמונה 10': 'תמונה 10',
            'תמונה 11': 'תמונה 11',
            'תמונה 12': 'תמונה 12',
            'תמונה 13': 'תמונה 13',
            'תמונה 14': 'תמונה 14',
            'תמונה\xa014': 'תמונה 14',
            'בר מצווה': 'בר מצווה',
            'מה אוהב לעשות': 'מה אוהב לעשות',
            'מה מעצבן': 'מה מעצבן',
            'משאלות לעתיד': 'משאלות לעתיד',
            'label': 'label',
            'בת מצווה': 'בר מצווה',
            'מה שאוהבת לעשות': 'מה אוהב לעשות',
            'מה שמעצבן': 'מה מעצבן',
            'מה שאוהב לעשות': 'מה אוהב לעשות',
            'מה שתרצה בעתיד': 'משאלות לעתיד',
            'יום הולדת 16': 'בר מצווה',
            'מה שמסב הנאה': 'מה אוהב לעשות',
            'מה שתרצי לעשות בעתיד': 'משאלות לעתיד',
            'תמונה1': 'תמונה 1',
            'מה אוהבת לעשות': 'מה אוהב לעשות',
            'הקלטה 9': 'תמונה 9',
            'הקלטה 10': 'תמונה 10',
            'הקלטה 11': 'תמונה 11',
            'הקלטה 12': 'תמונה 12',
            'הקלטה 13': 'תמונה 13',
            'הקלטה 14': 'תמונה 14',
            'מה שאוהבת': 'מה אוהב לעשות',
            'מה שרוצה לעשות בעתיד': 'משאלות לעתיד',
            'תמונה מספר 10': 'תמונה 10',
            'מה שתעשה בעתיד': 'משאלות לעתיד',
            'תמונה מספר 3': 'תמונה 3',
            'תמונה מספר 4': 'תמונה 4',
            '6': 'תמונה 6',
            'דברים שמעצבנים אותך': 'מה מעצבן',
            'לידת בן': 'בר מצווה',
            'מה תרצי לעשות בעתיד': 'משאלות לעתיד'
        }

    def _read_docx(self, file_path):
        """Read a .docx file and return its content as text."""
        doc = Document(file_path)
        return "\n".join([para.text for para in doc.paragraphs])

    def _map_question(self, question):
        """Map question to the predefined list using the mapping dictionary."""
        return self.question_mapping.get(question, question)

    def _extract_data(self, content):
        """Extract question names and corresponding text from the content."""
        pattern = re.compile(r'\+\+(.*?)\+\+')
        question_names = pattern.findall(content)
        sections = pattern.split(content)[1:]
        data = {}

        for name, text in zip(question_names, sections[1::2]):
            clean_name = name.strip()
            normalized_name = self._map_question(clean_name)
            if normalized_name in self.questions:
                data[normalized_name] = text.strip()

        return data

    def _load_files_from_folder(self, folder_path, label):
        """Load all .docx files from the given folder and extract data."""
        for file_name in os.listdir(folder_path):
            if file_name.endswith('.docx'):
                file_path = os.path.join(folder_path, file_name)
                content = self._read_docx(file_path)
                data = self._extract_data(content)
                data['label'] = label
                data['file_name'] = file_name  # Add file name to data
                self.data.append(data)

    def load_data(self):
        """Load data from control and patients folders."""
        control_folder = os.path.join(self.data_dir, 'control')
        patients_folder = os.path.join(self.data_dir, 'patients')

        self._load_files_from_folder(control_folder, label = 0)
        self._load_files_from_folder(patients_folder, label = 1)

    def get_dataframe(self):
        """Return the data as a pandas DataFrame."""
        df = pd.DataFrame(self.data, columns = self.questions + ['label', 'file_name'])
        return df


if __name__ == '__main__':
    # Usage example
    data_loader = DataLoader('data')  # Update with your data directory
    data_loader.load_data()
    df = data_loader.get_dataframe()
    df.to_csv("data.csv", index = False)

    # Display the DataFrame
    print(df.head())
