import urllib.request, re, json

url = 'https://www.humanizeai.pro/'
req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
html = urllib.request.urlopen(req).read().decode('utf-8', errors='ignore')

# Search for api endpoints or fetch calls
endpoints = set(re.findall(r'/(?:api|v1|humanize)[a-zA-Z0-9_\-/]*', html))
print('Endpoints in HTML:', endpoints)

# Find script tags
scripts = re.findall(r'src=["\']([^"\']+\.js[^"\']*)["\']', html)
print('Script URLs:')
for s in scripts[:10]:
    print(' ', s)
