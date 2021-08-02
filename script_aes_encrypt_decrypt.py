import pyaes, pbkdf2, binascii, os, secrets
import csv
import random
import string

iv = secrets.randbits(256)
passwordSalt = os.urandom(16)
password = "password"

print("IV:")
print(iv)
print("password:")
print(password)
print("passwordSalt:")
print(passwordSalt)

def encrypt(plaintext):
    key = pbkdf2.PBKDF2(password, passwordSalt).read(32)
    aes = pyaes.AESModeOfOperationCTR(key, pyaes.Counter(iv))
    ciphertext = aes.encrypt(plaintext)
    return binascii.hexlify(ciphertext).decode()

def decrypt(plaintext):
    key = pbkdf2.PBKDF2(password, passwordSalt).read(32)
    aes = pyaes.AESModeOfOperationCTR(key, pyaes.Counter(iv))
    ciphertext = aes.decrypt(binascii.unhexlify(plaintext))
    return ciphertext

def get_random_string(length):
    letters = string.ascii_letters + string.digits + string.punctuation
    result_str = ''.join(random.choice(letters) for i in range(length))
    return result_str


for j in range(1):
    new_csv_name = f'C:\\Users\\crist\\Documents\\AES tests\\tstnew4_output.csv'
    new_csv_file = open(new_csv_name, mode='w', encoding="utf8")
    new_csv_writer = csv.writer(new_csv_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)    
    new_csv_writer.writerow(['TEXT', 'ENCRYPTED'])
    iv = secrets.randbits(256)

    for i in range(1):
        plaintext = get_random_string(64)
        ciphertext = encrypt(plaintext)
        new_csv_writer.writerow([plaintext, ciphertext])
        print("Line: "+str(i))
    new_csv_file.close()