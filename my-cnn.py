strings = ['rock', 'pop', 'jazz', 'rock', 'pop', 'hiphop']
print(strings)

unique_strings = list(set(strings))
print(unique_strings)

str_to_int = {string: i for i, string in enumerate(unique_strings)}
print(str_to_int)

mapping = [str_to_int[string] for string in strings]
print(mapping)

int_to_str = {v: k for k, v in str_to_int.items()}
print(int_to_str)

reversed_mapping = [int_to_str[i] for i in mapping]
print(reversed_mapping)