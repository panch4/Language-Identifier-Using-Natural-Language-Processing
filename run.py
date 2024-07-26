import tkinter as tk
from tkinter import messagebox
import model
import dataset
import numpy as np

def word_to_hash(chars, word):
    word = word.lower()
    chars = list(chars)
    hashed = [chars.index(char) for char in word]
    while(len(hashed) < 10):
        hashed.append(-1)
    return np.ndarray((1,10), buffer=np.array(hashed), dtype=int)

def get_predicted_language(probs):
    languages = ["English", "Spanish", "Finnish", "Dutch", "Polish"]
    max_index = np.argmax(probs)
    return (probs[max_index], languages[max_index])

def on_enter(event):
    word = entry.get()
    if word == "":
        messagebox.showerror("Error", "Please enter a word.")
        return

    xs = data._encode(word_to_hash(chars, word), None, True)
    result = language_classifier.run(xs)
    probs = data._softmax(result.data)
    max_prob, pred_lang = get_predicted_language(probs[0])
    messagebox.showinfo("Prediction", "Predicted language is: {}\nWith a confidence of {:.2%}".format(pred_lang, max_prob))

language_classifier = model.LanguageClassificationModel()
data = dataset.LanguageClassificationDataset(language_classifier)
chars = data.chars
language_classifier.train(data)

test_predicted_probs, test_predicted, test_correct = data._predict('test')
test_accuracy = np.mean(test_predicted == test_correct)
print("Test set accuracy is: {:.2%}\n".format(test_accuracy))

root = tk.Tk()
root.title("Language Identifier")

label = tk.Label(root, text="Enter a word:")
label.pack()

entry = tk.Entry(root)
entry.pack()
entry.focus()

entry.bind("<Return>", on_enter)

root.mainloop()
