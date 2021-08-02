"""
# Encrypt the plaintext with the given key:
#   ciphertext = AES-256-CTR-Encrypt(plaintext, key, iv)
iv = secrets.randbits(256)
plaintext = "Text for encryption"
passwordSalt = os.urandom(16)
password = "s3cr3t*c0d3"
key = pbkdf2.PBKDF2(password, passwordSalt).read(32)
aes = pyaes.AESModeOfOperationCTR(key, pyaes.Counter(iv))
ciphertext = aes.encrypt(plaintext)
print('Encrypted:', binascii.hexlify(ciphertext))

# Decrypt the ciphertext with the given key:
#   plaintext = AES-256-CTR-Decrypt(ciphertext, key, iv)
aes = pyaes.AESModeOfOperationCTR(key, pyaes.Counter(iv))
decrypted = aes.decrypt(ciphertext)
print('Decrypted:', decrypted)
"""

import pyaes, pbkdf2, binascii, os, secrets
import csv
from base64 import b64encode, b64decode

#fixed
for i in range(1):
    new_csv_name = f'C:\\Users\\crist\\Documents\\AES tests\\tst{i}_output.csv'
    new_csv_file = open(new_csv_name, mode='w', encoding="utf8")
    new_csv_writer = csv.writer(new_csv_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    new_csv_writer.writerow(['TEXT', 'LENGHT', 'IV', "PASSWORD", "PASSWORDSALT", "ENCRYPTED"])
    iv = secrets.randbits(256)
    passwordSalt = os.urandom(16)
    password = "password"
    print("iv:")
    print(iv)
    print("password salt:")
    print(passwordSalt)

    with open('C:\\Users\\crist\\Documents\\AES tests\\Reviews.csv\\Reviews.csv', encoding="utf8") as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        err_count = 0
        for row in csv_reader:
            if line_count == 0:
                pass
            else:
                plaintext = bytes(row[9], 'utf-8').decode('utf-8', 'ignore')
                key = pbkdf2.PBKDF2(password, passwordSalt).read(32)
                aes = pyaes.AESModeOfOperationCTR(key, pyaes.Counter(iv))
                nobase64text = None
                ciphertext = None
                try:
                    nobase64text = aes.encrypt(plaintext)
                    ciphertext = b64encode(nobase64text).decode("utf-8")
                except:
                    err_count += 1
                    continue
                new_row = [plaintext, str(len(plaintext)), str(iv), str(password),  b64encode(passwordSalt).decode("utf-8"), ciphertext]
                new_csv_writer.writerow(new_row)
            line_count += 1
        print(err_count)
    new_csv_file.close()