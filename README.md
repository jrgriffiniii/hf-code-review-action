# hf-code-review-action
## GitHub Action for Automating Code Review using Hugging Face
_This is a fork of the GitHub project originally authored by (https://github.com/luiyen/llm-code-review)[https://github.com/luiyen/llm-code-review]. Aside from some trivial modifications, this is the labor of @luiyen, all credit and praise is due to them._

A container GitHub Action to review a pull request using Hugging Face large-language models.

If the size of a pull request is over the maximum chunk size of the Hugging Face API, the Action will split the pull request into multiple chunks and generate review comments for each chunk.
And then the Action summarizes the review comments and posts a review comment to the pull request.

## Pre-requisites
We have to set a GitHub Actions secret `HUGGINGFACEHUB_API_TOKEN` to use the Hugging Face API so that we securely pass it to the Action. Additionally, a secret `GH_TOKEN` must also be set in order to provide the permissions to post the comments on the Pull Request.

## Inputs

- `huggingFaceHubApiToken`: The Hugging Face API token generated to access the API.
- `ghToken`: The GitHub token to access the GitHub API.
- `githubRepository`: The GitHub repository to post a review comment.
- `githubPullRequestNumber`: The GitHub pull request number to post a review comment.
- `gitCommitHash`: The git commit hash to post a review comment.
- `pullRequestDiffs`: A directory containing one or more `.diff` files generated from the Pull Request being reviewed.
- `pullRequestDiffChunkSize`: The chunk size of the diff of the pull request to generate a review comment.
- `repoId`: The Hugging Face model repository ID.
- `temperature`: The temperature for the Hugging Face model.
- `topP`: The top_p for the Hugging Face model.
- `topK`: The top_k for the Hugging Face model.
- `maxNewTokens`: The max_tokens for the Hugging Face model.
- `logLevel`: The logging verbosity level (for troubleshooting).

As you might know, a model of Hugging Face has limitation of the maximum number of input tokens.
So we have to split the diff of a pull request into multiple chunks, if the size of the diff is over the limitation.
We can tune the chunk size based on the model we use.

## Example usage
Here is an example to use the Action to review a pull request of the repository.
The actual file is located at [`.github/workflows/test-action.yml`](.github/workflows/test-action.yml).


```yaml
name: "Test the Hugging Face Code Review"

on:
  pull_request:
    types:
      - open
      - synchronize
      - ready_for_review
    paths-ignore:
      - "*.md"
      - "LICENSE"

jobs:
  review:
    runs-on: ubuntu-latest
    permissions:
      contents: read
      pull-requests: write
    steps:
      - uses: actions/checkout@v3
      - name: "Get diff of the pull request"
        id: get_diff
        shell: bash
        env:
          DEFAULT_BRANCH: "${{ github.event.repository.default_branch }}"
          PULL_REQUEST_HEAD_REF: "${{ github.event.pull_request.head.ref }}"
        run: |-
          # Fetch the default branch
          git fetch origin "${{ env.DEFAULT_BRANCH }}"
          # Include only source code files for the diff
          git diff "origin/${{ env.DEFAULT_BRANCH }}" --name-only > files.diff
          IDX=0
          while read GIT_DIFF_FILE; do
            echo $GIT_DIFF_FILE
            git diff "origin/${{ env.DEFAULT_BRANCH }}" $GIT_DIFF_FILE > .diffs/$IDX.diff
            ((IDX+=1))
          done < files.diff
      - uses: ./
        name: "Hugging Face Code Review"
        id: hf_review
        with:
          huggingFaceHubApiToken: ${{ secrets.HUGGINGFACEHUB_API_TOKEN }}
          ghToken: ${{ secrets.GH_TOKEN }}
          githubRepository: ${{ github.repository }}
          githubPullRequestNumber: ${{ github.event.pull_request.number }}
          gitCommitHash: ${{ github.event.pull_request.head.sha }}
          repoId: "mistralai/Mistral-7B-Instruct-v0.2"
          temperature: "0.1"
          maxNewTokens: "250"
          topK: "10"
          topP: "0.95"
          pullRequestDiffs: "./.diffs"
          pullRequestChunkSize: "3500"
          logLevel: "DEBUG"
```
