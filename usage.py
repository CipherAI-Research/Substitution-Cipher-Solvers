from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
from string import ascii_lowercase
import Levenshtein
import random

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = AutoTokenizer.from_pretrained("Cipher-AI/Substitution-Cipher-Alphabet-Eng")
alphabet_model = AutoModelForSeq2SeqLM.from_pretrained("Cipher-AI/Substitution-Cipher-Alphabet-Eng").to(device)
correction_model = AutoModelForSeq2SeqLM.from_pretrained("Cipher-AI/AutoCorrect-EN-v2").to(device)

def similarity_percentage(s1, s2):
    distance = Levenshtein.distance(s1, s2)

    max_len = max(len(s1), len(s2))

    similarity = (1 - distance / max_len) * 100

    return similarity

def decode(cipher_text, key):
  decipher_map = {ascii_lowercase[i]: j for i, j in enumerate(key[:26])}
  decipher_map.update({ascii_lowercase[i].upper(): j.upper() for i, j in enumerate(key[:26])})
  ans = ''.join(map(lambda x: decipher_map[x] if x in decipher_map else x, cipher_text))
  return ans

def model_pass(model, input, max_length=256):
  inputs = tokenizer(input, return_tensors="pt", padding=True, truncation=True, max_length=256).to(device)
  outputs = model.generate(inputs["input_ids"], max_length=max_length)
  result = tokenizer.decode(outputs[0], skip_special_tokens=True)
  return result

def decipher(cipher_text, key) -> str:
  decipher_map = {ascii_lowercase[i]: j for i, j in enumerate(key[0])}
  decipher_map.update({ascii_lowercase[i].upper(): j.upper() for i, j in enumerate(key[0])})

  result = ''.join(map(lambda x: decipher_map[x] if x in decipher_map else x, cipher_text[0]))

  return result

def cipher(plain_text) -> tuple[str, list]:
  alphabet_map = list(ascii_lowercase)
  random.shuffle(alphabet_map)
  alphabet_map = {i : j for i, j in zip(ascii_lowercase, alphabet_map)}

  alphabet_map.update({i.upper() : j.upper() for i, j in alphabet_map.items()})

  cipher_text = ''.join(map(lambda x: alphabet_map[x] if x in alphabet_map else x, plain_text))
  return cipher_text, alphabet_map

def correct_text(cipher_text, model_output):
  cipher_text = cipher_text.split(' ')
  model_output = model_output.split(' ')

  letter_map = {i: {j: 0 for j in ascii_lowercase} for i in ascii_lowercase}


  # Levenstein distance for lenghts of words
  n = len(cipher_text)
  m = len(model_output)

  i = 0
  j = 0
  dp = [[0 for _ in range(m + 1)] for _ in range(n + 1)]

  for i in range(n + 1):
    dp[i][0] = i


  for j in range(m + 1):
    dp[0][j] = j

  for i in range(1, n + 1):
    for j in range(1, m + 1):
      if len(cipher_text[i - 1]) == len(model_output[j - 1]):
        dp[i][j] = dp[i - 1][j - 1]

      else:
        dp[i][j] = min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1]) + 1

  i = n
  j = m
  while i > 0 and j > 0:

    before = min([(0, dp[i - 1][j - 1]), (1, dp[i - 1][j]), (2, dp[i][j - 1])], key=lambda x: x[1])
    match before[0]:
      case 0:
        if dp[i - 1][j - 1] == dp[i][j]:
          # If the same we add them to letter map
          cipher = cipher_text[i-1]
          model_o = model_output[j-1]

          for c_letter, m_letter in zip(cipher.lower(), model_o.lower()):
            if c_letter in letter_map and m_letter in letter_map[c_letter]:
              letter_map[c_letter][m_letter] += 1

        i = i - 1
        j = j - 1
      case 1:
        i = i - 1
      case 2:
        j = j - 1

  for letter in ascii_lowercase:
    letter_sum = sum(letter_map[letter].values())
    if letter_sum == 0:
      # That letter wasn't in the text
      letter_map[letter] = None
      continue

    # Sorted from most accuring to least
    letter_map[letter] = [(k, v / letter_sum) for k, v in sorted(letter_map[letter].items(), key=lambda item: item[1], reverse=True)]

  change_map = {
      i : None for i in ascii_lowercase
  }

  for i in range(len(ascii_lowercase)):
    for letter in ascii_lowercase:
      if letter_map[letter] is None:
        continue  # That letter wasn't in the text

      # If None then it didn't get substituted earlier
      map_letter = letter_map[letter][i][0]
      if (letter_map[letter][i][1] > 0 and (change_map[map_letter] is None
          or (change_map[map_letter][2] < letter_map[letter][i][1] and change_map[map_letter][1] >= i))):
        change_map[map_letter] = (letter, i, letter_map[letter][i][1])
        # Letter, iteration, percentage

  change_map = {i[1][0]: i[0] for i in change_map.items() if i[1] is not None}

  for letter in ascii_lowercase:
    if letter not in change_map:
      change_map[letter] = '.'


  # Add uppercases
  change_map.update(
    {
      i[0].upper() : i[1].upper() for i in change_map.items()
    }
  )

  new_text = []
  for cipher in cipher_text:
    new_word = ""
    for c_letter in cipher:
      if c_letter in change_map:
        new_word += change_map[c_letter]

      else:
        new_word += c_letter


    new_text.append(new_word)

  return ' '.join(new_text)

def crack_sub(cipher_text):
  output = model_pass(alphabet_model, cipher_text, 26)
  decoded = decode(cipher_text, output)
  second_pass = model_pass(correction_model, decoded, len(decoded))
  second_text = correct_text(cipher_text, second_pass)
  third_pass = model_pass(correction_model, second_text, len(decoded))

  return third_pass

"""
Use crack_sub() function to solve monoalphabetic substitution ciphers!
"""
