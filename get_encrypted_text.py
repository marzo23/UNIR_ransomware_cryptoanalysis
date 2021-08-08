import re
import csv

docs_path = "C:\\Users\\crist\\Documents\\AES tests\\"
encrypted_path = f'{docs_path}\\AES tests\\enc\\'
plaintext_path = f'{docs_path}\\AES tests\\dec\\' 

with open('output.txt', encoding="utf8") as output_file:
    with open('output.csv', encoding="utf8") as csv_file:
        new_csv_writer = csv.writer(csv_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        new_csv_writer.writerow(['TEXT', "ENCRYPTED"])
        for line in output_file:
            match = re.match(f'Renaming {docs_path}(.*)to(.*)', line)
            if match and len(match) == 2:
                decrypted_file_path = plaintext_path + match[0]
                encrypted_file_path = plaintext_path + match[1] + ".encrypted"
                row = []
                with open(encrypted_file_path, encoding="utf8") as encrypted_file:
                    row.append("\n".join(encrypted_file.readlines()))
                with open(decrypted_file_path, encoding="utf8") as decrypted_file:
                    row.append("\n".join(decrypted_file.readlines()))
                new_csv_writer.writerow(row)


