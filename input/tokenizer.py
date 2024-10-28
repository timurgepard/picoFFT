import pickle
import re
import contractions
from word2number import w2n
from num2words import num2words

def process_token3(token):
    # Check if the token contains a punctuation sign
    if any(c in ' ' for c in token):

        matches1 = re.match(r'([\s]+)([a-zA-Z\s])', token)
        matches2 = re.match(r'([a-zA-Z]+)([\s])', token)

        if matches1: return [matches1.group(1), matches1.group(2)]
        if matches2: return [matches2.group(1), matches2.group(2)]

    return [token]

def process_tokens3(tokens):
    result = []
    for token in tokens:
        # Process each token
        result.extend(process_token3(token))
    return result


def consolidate_whitespace(tokens):
    result = []
    prev_token = None

    for token in tokens:
        # Check if the current tokenacter is a space and the previous one was also a space
        if token == ' ' and prev_token[-1] == ' ':
            continue  # Skip consecutive spaces
        else:
            result.append(token)
            prev_token = token

    return result


def process_token2(token):
    # Check if the token contains a punctuation sign
    if any(c in '!:?.,-("*' for c in token):
        if token == "-.": token = ". "
        # Use regular expressions to split the token into letter and whitespace parts
        matches = re.match(r'([!:?.,\-\(\"\*]+)([a-zA-Z\s])', token)
        if matches:
            #print(matches.group(1), matches.group(2))
            punctuation = ' ' + matches.group(1) + ' '
            letter = matches.group(2)


            return [punctuation, letter]
        else:
            return [token]
    else:
        return [token]

def process_tokens2(tokens):
    result = []
    for token in tokens:
        # Process each token
        result.extend(process_token2(token))
    return result


def process_token(token):
    # Check if the token contains a punctuation sign
    if any(c in '.,;:?!-)' for c in token):
        
        # Use regular expressions to split the token into letter and whitespace parts
        matches = re.match(r'([a-zA-Z\s]+)([.,:?!\-\)])', token)
        if matches:
            #print(matches.group(1), matches.group(2))
            letter = matches.group(1) + ' ' if matches.group(1) != ' ' else ' '
            punctuation = ' ' + matches.group(2) + ' '
            return [letter, punctuation]
        else:
            return [token]
    else:
        return [token]

def process_tokens(tokens):
    result = []
    for token in tokens:
        # Process each token
        result.extend(process_token(token))
    return result

def remove_short_lines(text):
    pattern = re.compile(r'^.{6,}$', re.MULTILINE)

    # Use the pattern to find all matching lines
    matches = pattern.findall(text)

    # Join the matching lines back into a single string
    return '\n'.join(matches)

def remove_symbols_before_colon(text):
    # Define the regular expression pattern
    pattern = re.compile(r'(?<=[^a-zA-Z]):')

    # Use the sub() function to replace matches with an empty string
    result = pattern.sub('', text)

    return result

def add_space(text):
    text = re.sub(r'([^\s])\s*([\(])', r'\1 \2', text)
    return re.sub(r'([).,;:!?])([^\s])', r'\1 \2', text)

def remove_chapter_lines(text):
    # Define the regular expression pattern
    pattern = re.compile(r'^\s*Chapter.*?$\n^(.{0,100})$\n', re.MULTILINE)

    # Use the sub function to remove matching lines
    result = re.sub(pattern, '', text)

    return result

def remove_chapter(text):
    # Define the regular expression pattern
    pattern = re.compile(r'^\s*Chapter.*?$\n', re.MULTILINE)

    # Use the sub function to remove matching lines
    result = re.sub(pattern, '', text)

    return result

def remove_content_lines(text):
    # Define the regular expression pattern
    pattern = re.compile(r'^\s*Content.*?$\n^(.{0,300})$\n', re.MULTILINE)

    # Use the sub function to remove matching lines
    result = re.sub(pattern, '', text)

    return result


def replace_numbers_with_text(text):
    words = text.split()
    updated_words = []

    for word in words:
        try:
            # Try to convert the word to a number
            number = w2n.word_to_num(word)
            # Convert the number to words and append to the updated list
            updated_words.append(num2words(number))
        except:
            # If the word is not a number, keep it as is
            updated_words.append(word)

    # Join the updated words to form the modified text
    updated_text = ' '.join(updated_words)
    
    return updated_text

def remove_numbers(text):
    # Use regular expression to remove all numeric characters
    text_without_numbers = re.sub(r'\d+', '', text)
    return text_without_numbers

def remove_mentions_and_tags(text):
    #text = re.sub(r'(.*?)', '', text)
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub(r'\<.*?\>', '', text)
    text = re.sub(r'^\s*\,', '', text, flags=re.MULTILINE)
    text = re.sub(r'^\s*\.', '', text, flags=re.MULTILINE)
    text = re.sub(r'@\S*', '', text)
    text = re.sub(r'#\S*', '', text)
    return re.sub(r'&\S*', '', text)

def expand_contractions(text):
    expanded_words = [] 
    for word in text.split():
       expanded_words.append(contractions.fix(word)) 
    return ' '.join(expanded_words)

def convert_to_lowercase(text):
    # Use a regular expression to find all uppercase letters
    pattern = re.compile(r'[A-Z]')
    
    # Replace each uppercase letter with its lowercase equivalent
    result = re.sub(pattern, lambda x: x.group().lower(), text)
    
    return result

def remove_url(text):
 return re.sub(r'https?:\S*', '', text)

def remove_new_line_punc(text):
    return re.sub( r'(?<=\n)[\W]', '', text)

with open('./input/input1.txt', 'r', encoding='utf-8') as f:
    text1 = f.read()

with open('./input/input2.txt', 'r', encoding='utf-8') as f:
    text2 = f.read()

with open('./input/input3.txt', 'r', encoding='utf-8') as f:
    text3 = f.read()

with open('./input/input4.txt', 'r', encoding='utf-8') as f:
    text4 = f.read()


text = text1 + ' /n' + text2 + ' /n' + text3 + ' /n' + text4 + ' /n'

text = text.encode("ascii", "ignore")
text = text.decode()


text = remove_chapter_lines(text)
text = remove_chapter(text)
text = remove_content_lines(text)
text = remove_url(text)
text = remove_mentions_and_tags(text)


#text = replace_numbers_with_text(text)
#text = remove_numbers(text)



for _ in range(1):
    text = text.replace('<', ' ')
    text = text.replace('>', ' ')
    text = text.replace('{', '(')
    text = text.replace('}', ')')
    text = text.replace(']', ')')
    text = text.replace('[', '(')
    text = text.replace('^', ' ')
    text = text.replace('|', ' ')
    text = text.replace('+', ' ')
    text = text.replace('*', ' ')
    text = text.replace('~', ' ')
    text = text.replace('_', ' ')
    text = text.replace('=', ' ')
    text = text.replace('$', ' ')
    text = text.replace('/', ' ')
    text = text.replace('\\', ' ')
    text = text.replace(';', '.')
    text = text.replace('`', "'")
    text = text.replace('(', ',')
    text = text.replace(')', ',')
    text = remove_new_line_punc(text)


for _ in range(7):

    text = text.replace('(,', '(')
    text = text.replace('?,', '?')
    text = text.replace('?.', '?')
    text = text.replace('.,', '.')
    text = text.replace(',.', '.')
    text = text.replace(' )', ')')
    text = text.replace('.)', ')')
    text = text.replace(' ,', ',')
    text = text.replace(',-', ',')
    text = text.replace(', -', ',')
    text = text.replace('-,', ',')
    text = text.replace('- ,', ',')
    text = text.replace('?-', '? ')
    text = text.replace('-)', ')')
    text = text.replace('-?', '?')
    text = text.replace('(?', '(')
    text = text.replace('(!', '(')
    text = text.replace('(-', '(')
    text = text.replace(')-', ') ')
    text = text.replace('.!', '!')
    text = text.replace('!.', '!')
    text = text.replace('! -', '! ')
    text = text.replace('-.', '.')
    text = text.replace('- .', '.')
    text = text.replace('.-', '.')
    text = text.replace('. -', '.')
    text = text.replace('::', ':')
    text = text.replace('(.', '(')
    text = text.replace('(:', '(')
    text = text.replace(':)', ')')
    text = text.replace(':.', '.')
    text = text.replace('-:', ':')
    text = text.replace(':-', ':')
    text = text.replace('( )', '')
    text = text.replace('()', '')
    text = text.replace('!!', '!')
    text = text.replace('-!', '!')
    text = text.replace('!-', '!')
    text = text.replace(',!', '!')
    text = text.replace('!,', '!')
    text = text.replace('!.', '!')
    text = text.replace('.!', '!')
    text = text.replace('. .', '.')
    text = text.replace(', ,', ',')
    text = text.replace(', .', '.')
    text = text.replace('. ,', '.')
    text = text.replace('((', '(')
    text = text.replace('))', ')')
    text = text.replace('??', '?')
    text = text.replace('  ', ' ')
    text = text.replace("--", "-")
    text = text.replace("''", "'")
    text = text.replace(';;', ';')
    text = text.replace(',,', ',')
    text = text.replace('..', '.')
    text = text.replace('\n\n', '\n')
    text = text.replace('\n\r', '\r')
    text = text.replace('\n', '\r')
    text = text.replace('\r\r', '\r')



text = text.replace('\r', ' * ')
text = text.replace('" ', '@')
text = text.replace(' "', '@')
text = text.replace('"', '@')
text = text.replace('@', ' " ')




for _ in range(7):
    text = text.replace('  ', ' ')
    text = text.replace('-.', '.')

text = expand_contractions(text)
text = convert_to_lowercase(text)
text = text.replace("'", "")
text = add_space(text)
text = remove_symbols_before_colon(text)


text = text + ' ' * (2 - (len(text) % 2))  # Make sure the text length is divisible by 2 by adding spaces if needed



text_tokens = [text[i:i + 2] for i in range(0, len(text), 2)]
text_tokens = process_tokens(text_tokens)
text_tokens = process_tokens2(text_tokens)
text_tokens = process_tokens3(text_tokens)
text_tokens = consolidate_whitespace(text_tokens)

tokens = sorted(list(set(text_tokens)))

with open('./input/tokens.pkl', 'wb+') as f:
    pickle.dump(tokens, f)

with open('./input/input.pkl', 'wb+') as f:
    pickle.dump(text_tokens, f)

