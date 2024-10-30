import click
import pandas as pd
import numpy as np
from nltk import word_tokenize
import collections
import heapq

@click.group()
def cli():
    pass

@click.command()
@click.argument('dataset')
def start(dataset):
    click.echo('Parsing dataset...')
    data = pd.read_csv(dataset)

    click.echo('Loading tweets...')

    # [ (tweet, retweet_count)]
    tweets = []
    for tweet in data.iloc:
        tweets.append((tweet['text'], int(tweet['retweet_count']) if not np.isnan(tweet['retweet_count']) else 0))

    # {
    #     token: count
    # }
    click.echo('Getting word counts...')
    word_counts = collections.defaultdict(int)
    for tweet, _ in tweets:
        for token in word_tokenize(tweet):
            if len(token) <= 3:
                continue
            token = token.lower()
            word_counts[token] += 1
    
    # {
    #     token: [(retweet, index)]
    # }
    click.echo('Building search index...')
    search_index = {}
    for i, (tweet, retweet_count) in enumerate(tweets):
        for token in word_tokenize(tweet):
            if len(token) <= 3:
                continue
            token = token.lower()
            search_index[token] = search_index.get(token, [])
            search_index[token].append((retweet_count, i))
    for token, tweet_indexes in search_index.items():
        search_index[token] = sorted(tweet_indexes, key=lambda x: x[0], reverse=True)


    # {
    #     letter: {
    #         letter:...,
    #         top_ten: [ (count, word) ],
    #         is_token: bool
    #     }
    # }
    click.echo('Constructing autocomplete trie...')
    trie = {}
    for token in search_index:
        node = trie
        for i, char in enumerate(token):
            node[char] = node.get(char, { 'top_ten': [], 'tweets': [] })

            heapq.heappush(node[char]['top_ten'], (word_counts[token], token))
            if len(node[char]['top_ten']) > 10:
                heapq.heappop(node[char]['top_ten'])

            if i == len(token) - 1:
                node[char]['is_token'] = True
            
            node = node[char]


    click.echo('')
    click.echo('--- Welcome to 2015 Resolutions ---')
    click.echo('Search for a tweet:')
    while True:
        word = ''
        while True:
            char = click.getchar().lower()
            if char == '\x0d': # enter
                break
            elif char == '\x08': # backspace
                word = word[:-1]
            elif char.isalnum():
                word += char

            # display search suggestions
            click.clear()
            click.echo('Search for a tweet:')
            click.echo(word)
            click.echo('\n--- suggested words ---')
            if word:
                node = trie
                for c in word:
                    if c in node:
                        node = node[c]
                    else:
                        node = None
                        break
                suggestions = sorted(node['top_ten'] if (node and node['top_ten']) else [], key=lambda x: x[0], reverse=True)
                    
                for count, suggestion in suggestions:
                    click.echo(f"{suggestion} - {count}")

        # display search results
        click.clear()
        click.echo('Search for a tweet:')
        click.echo(word)
        click.echo('\n--- tweets ---')
        tweet_indexes = search_index.get(word, [])
        if not tweet_indexes:
            click.echo(f'No tweets with "{word}" were found. Try another search?')
        for _, tweet_index in tweet_indexes[:10]:
            click.echo(f"{tweets[tweet_index][0]}\n--")
        click.echo('')

cli.add_command(start)

if __name__ == '__main__':
    cli()