#import markdown
import os
import time
import re


# FILENAME = 'README.md'
# FILENAME = 'transformer.md'
#FILENAME = 'article.md'
FILENAME = 'heuristic.md'
size = os.get_terminal_size()
MAX_WORD_SIZE = size[1]


def parse_file_to_clean_html(file_name):
    with open(file_name, 'r', encoding='utf8') as file:
        file_text = file.read()
        #html_text = markdown.markdown(data)
        return file_text

def turn_into_document(file_text):
    document = []
    lines = re.split('\n', file_text)
    for line in lines:
        #print(f"Line: -{line}-")
        if line != "":
            if line[0] == '#':
                document.append(line)
                document.append('\n')
            else:
                #print('test1')
                words = line.split()
                #print(words)
                #cleaned_line = [re.sub(r'<.*?>', '', word) for word in words]
                #print(cleaned_line)
                for word in words:
                    document.append(word)
                document.append('\n')
            #print('test2')
    
    return document


def print_out_document(document):
    max_word_length = MAX_WORD_SIZE
    sleep_time_dict = {
        0: 0.18,
        1: 0.5
    }
    sleep_time = 0

    for word in document:
        
        if len(word) > max_word_length:
            max_word_length = len(word)

        print("\r" + " " * max_word_length + "\r", end="")
        if word != "\n":
            print(f"\r{word}", end="")
        #sleep_time = 1 if (word[0] == '#' or word == "\n") else 0
        #time.sleep(sleep_time_dict[sleep_time])
        # mean time of words in any language is 68ms
        sleep_time = len(word) * 0.068
        if sleep_time < 0.1:
            sleep_time = 0.5
        time.sleep(sleep_time)
        


def main():
    file_text = parse_file_to_clean_html(FILENAME)
    document = turn_into_document(file_text)
    print_out_document(document)


if __name__ == "__main__":
    main()

    

    

    


