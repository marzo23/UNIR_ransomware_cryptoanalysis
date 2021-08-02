import random
import string

def get_random_string(length):
    letters = string.ascii_letters + string.digits + string.punctuation
    result_str = ''.join(random.choice(letters) for i in range(length))
    return result_str

for i in range(750000):
    new_file_name = f'C:\\Users\\crist\\Documents\\AES tests\\plaintext{i}.txt'
    with open(new_file_name, mode='w', encoding="utf8") as new_file:
        plaintext = get_random_string(64)
        new_file.write(plaintext)