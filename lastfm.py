import requests

with open('lastfm_api_key.txt') as f:
    lastfm_api_key = f.read()
    
    
def ask_lastfm_similar_artists(artist):
    url = 'http://ws.audioscrobbler.com/2.0/'

    data = {'method': 'artist.getsimilar', 
            'artist': artist, 
            'autocorrect': 0,
            'api_key': lastfm_api_key, 
            'format': 'json'}
    
    response = requests.post(url, data=data)
    
    if response.status_code == 200:
        r = response.json()
        similar_list = r['similarartists']['artist']
        return similar_list
    
    print('Sorry, some kind of error with status code {}'.format(response.status_code))
    return None
    
def print_lastfm_response(similar_list):
    print('# of similar artists: {:,}'.format(len(similar_list)))
    for sim in similar_list:
        print(sim['match'], sim['name'])  # match is b/w 0 and 1 (float)
        
def parse_lastfm_response(similar_list):
    list_ = []
    for sim in similar_list:
        list_.append((sim['name'], float(sim['match'])))  # match is b/w 0 and 1 (float)
    return list_
    
def get_top_n_similar_artists(artist, n=5, match_min=0, include_src=False):
    similar_list = ask_lastfm_similar_artists(artist)    
    if similar_list is None:
        return

    list_ = parse_lastfm_response(similar_list)
    output = []
    for i, name_match in enumerate(list_):
        name, match = name_match
        if (i < n) and (match >= match_min):
            if include_src:
                output.append((artist, name))
            else:
                output.append(name)
    return output
    
def get_top_n_linked_artists(artist, n_min=50, match_min=0.5, n_per_artist=5, include_src=False):
    if include_src:
        output = [(artist, artist)]
    else:
        output = [artist]
        
    queue = [artist]
    while True:
        if len(queue) == 0:
            print('Couldn\'t find {:,} artists...'.format(n_min))
            break
        
        focus = queue[0]
        list_ = get_top_n_similar_artists(focus, n=n_per_artist, match_min=match_min, include_src=False)
        
        to_add = []
        for elem in list_:  # Check if it's already checked/added
            if include_src:
                if elem.lower() in [o.lower() for _, o in output]:
                    continue
                to_add.append((focus, elem))
            else:
                if elem.lower() in [o.lower() for o in output]:
                    continue
                to_add.append(elem)
        
        output.extend(to_add)
        if len(output) >= n_min:
            break
        
        if include_src:
            queue.extend([elem for _, elem in to_add])
        else:
            queue.extend(to_add)       
        queue = queue[1:]  # Remove current artist
            
    return output
    