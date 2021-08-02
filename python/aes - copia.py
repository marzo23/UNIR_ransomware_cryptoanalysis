import pyaes, pbkdf2, binascii, os, secrets

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


import csv
from base64 import b64encode, b64decode

#fixed
for i in range(1):
    new_csv_name = f'C:\\Users\\crist\\Documents\\AES tests\\tst{i}_output.csv'
    new_csv_file = open('employee_file.csv', mode='w')
    new_csv_writer = csv.writer(new_csv_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    new_csv_writer.writerow(['TEXT', 'LENGHT', 'IV', "PASSWORD", "PASSWORDSALT", "ENCRYPTED"])
    iv = secrets.randbits(256)
    passwordSalt = os.urandom(16)
    password = "password"
    print("iv:")
    print(iv)
    print("password salt:")
    print(passwordSalt)

    with open('C:\\Users\\crist\\Documents\\AES tests\\Reviews.csv\\Reviews.csv') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        for row in csv_reader:
            """
            if line_count < 2:
                print("\n\n")
                plaintext = "IhaveboughtseveraloftheV" #row[9][:1]
                key = pbkdf2.PBKDF2(password, passwordSalt).read(32)
                aes = pyaes.AESModeOfOperationCTR(key, pyaes.Counter(iv))
                ciphertext = aes.encrypt(plaintext)
                print('Encrypted:', binascii.hexlify(ciphertext))
                print('plain text:', plaintext)
                key = pbkdf2.PBKDF2(password, passwordSalt).read(32)
                aes = pyaes.AESModeOfOperationCTR(key, pyaes.Counter(iv))
                decrypted = aes.decrypt(ciphertext)
                print('Decrypted:', decrypted)
            else:
                break
            """
            if line_count == 0:
                pass
            else:
                #plaintext = "IhaveboughtseveraloftheV" 
                plaintext = row[9]
                key = pbkdf2.PBKDF2(password, passwordSalt).read(32)
                aes = pyaes.AESModeOfOperationCTR(key, pyaes.Counter(iv))
                ciphertext = b64encode(aes.encrypt(plaintext)).decode("utf-8")
                new_row = [plaintext, str(len(plaintext)), str(iv), str(password),  b64encode(passwordSalt).decode("utf-8"), ciphertext]
                new_csv_writer.writerow(new_row)
                if line_count < 2:
                    print("\n\n")
                    print("\t".join(new_row))
                    key2 = pbkdf2.PBKDF2(str(password), b64decode(b64encode(passwordSalt).decode("utf-8"))).read(32)
                    aes2 = pyaes.AESModeOfOperationCTR(key2, pyaes.Counter(int(str(iv))))
                    decrypted = aes2.decrypt(b64decode(ciphertext))
                    print("\n\n OUTPUT", decrypted)
                else:
                    break
            line_count += 1

    new_csv_file.close()