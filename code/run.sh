docker build -t 'latest' .

docker run -v /Users/tearth/Documents/GitHub/ltfs_hack/incident-insights/topcoder_cognitive_state/data/:/work/data/ \
           -v /Users/tearth/Documents/GitHub/ltfs_hack/incident-insights/topcoder_cognitive_state/sample-submission/code/model/:/work/model/ \
           latest sh train.sh /work/data/data_training.zip

docker run -v /Users/tearth/Documents/GitHub/ltfs_hack/incident-insights/topcoder_cognitive_state/data/:/data/ \
           -v /Users/tearth/Documents/GitHub/ltfs_hack/incident-insights/topcoder_cognitive_state/sample-submission/code/model/:/work/model/ \
           latest sh test.sh /data/data_provisional.zip /data/solution.csv 