import os
import random
import re
import sys

DAMPING = 0.85
SAMPLES = 10000


def main():
    if len(sys.argv) != 2:
        sys.exit("Usage: python pagerank.py corpus")
    corpus = crawl(sys.argv[1])
    ranks = sample_pagerank(corpus, DAMPING, SAMPLES)
    print(f"PageRank Results from Sampling (n = {SAMPLES})")
    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")
    ranks = iterate_pagerank(corpus, DAMPING)
    print(f"PageRank Results from Iteration")
    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")


def crawl(directory):
    """
    Parse a directory of HTML pages and check for links to other pages.
    Return a dictionary where each key is a page, and values are
    a list of all other pages in the corpus that are linked to by the page.
    """
    pages = dict()

    # Extract all links from HTML files
    for filename in os.listdir(directory):
        if not filename.endswith(".html"):
            continue
        with open(os.path.join(directory, filename)) as f:
            contents = f.read()
            links = re.findall(r"<a\s+(?:[^>]*?)href=\"([^\"]*)\"", contents)
            pages[filename] = set(links) - {filename}

    # Only include links to other pages in the corpus
    for filename in pages:
        pages[filename] = set(
            link for link in pages[filename]
            if link in pages
        )

    return pages


def transition_model(corpus, page, damping_factor):
    """
    Return a probability distribution over which page to visit next,
    given a current page.

    With probability `damping_factor`, choose a link at random
    linked to by `page`. With probability `1 - damping_factor`, choose
    a link at random chosen from all pages in the corpus.
    """
    if corpus[page]:
        prob_distribution = {page: (1 - damping_factor)/len(corpus) for page in corpus}
        for link in corpus[page]:
            prob_distribution[link] += damping_factor*1/len(corpus[page])
    else:
        prob_distribution = {page: 1/len(corpus) for page in corpus}
        
    return prob_distribution
        


def sample_pagerank(corpus, damping_factor, n):
    """
    Return PageRank values for each page by sampling `n` pages
    according to transition model, starting with a page at random.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """
    pageCounts={page: 0 for page in corpus}
    start = random.choice(list(corpus.keys()))
    pageCounts[start] += 1
    prob_distribution = transition_model(corpus, start, damping_factor)
    for _ in range(1,n):
        newPage=random.choices(list(prob_distribution.keys()), weights=list(prob_distribution.values()), k=1)[0]
        prob_distribution=transition_model(corpus, newPage, damping_factor)
        pageCounts[newPage] += 1
    
    page_ratio = {page: pageCounts[page]/n for page in pageCounts}
    return page_ratio
        


def iterate_pagerank(corpus, damping_factor):
    """
    Return PageRank values for each page by iteratively updating
    PageRank values until convergence.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """
    N=len(corpus)
    def _iter(PR):
        new_PR = {page: (1-damping_factor)/N for page in PR}
        for i in corpus:
            for p in corpus[i]:
                new_PR[p]+=damping_factor*PR[i]/len(corpus[i])
                
        distance=max(abs(new_PR[k] - PR[k]) for k in PR)
        if distance < 1e-3:
            return new_PR
        else:
            return _iter(new_PR)
        
    initial_PR = {page: 1/len(corpus) for page in corpus}
    return _iter(initial_PR)
        


if __name__ == "__main__":
    main()
