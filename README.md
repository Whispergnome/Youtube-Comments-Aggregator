# Youtube-Comments-Aggregator
This project aims at taking a single youtube's video's comments and group them up by comment's meaning.

The first part is the yt_fetch_comments.py which needs an API KEY, reach a single youtube video, and creates a .csv file of raw comments.
The second part is cluster_comments.py which takes .csv file and attempts to cluster similar comments.

run_pipeline.py runs both one after the other.
