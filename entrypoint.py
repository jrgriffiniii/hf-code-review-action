#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#          http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

import json
import os
from typing import List, Optional

import click
import requests
from loguru import logger

from langchain_community.chat_models.huggingface import ChatHuggingFace
from langchain_community.llms.huggingface_endpoint import HuggingFaceEndpoint
from langchain_community.embeddings import HuggingFaceHubEmbeddings
from langchain_core.output_parsers import StrOutputParser
from langchain.prompts import ChatPromptTemplate
from transformers import AutoTokenizer


def check_required_env_vars():
    """Check required environment variables"""
    required_env_vars = [
        "HUGGINGFACEHUB_API_TOKEN",
        "GH_TOKEN",
        "GITHUB_REPOSITORY",
        "GITHUB_PULL_REQUEST_NUMBER",
        "GIT_COMMIT_HASH",
    ]
    for required_env_var in required_env_vars:
        if os.getenv(required_env_var) is None:
            raise ValueError(f"{required_env_var} is not set")


class PullRequest:

    def __init__(self, github_token: str, repository: str, number: int):

        self._github_token = github_token
        self._repository = repository
        self._number = number

    def comment(self, commit_hash: str, body: str):
        """Create a comment to a pull request"""

        headers = {
            "Accept": "application/vnd.github.v3.patch",
            "authorization": f"Bearer {self._github_token}",
        }

        params = {"commit_id": commit_hash, "body": body, "event": "COMMENT"}

        url = f"https://api.github.com/repos/{self._repository}/pulls/{self._number}/reviews"
        data = json.dumps(params)
        response = requests.post(url, headers=headers, data=data, timeout=3.0)

        return response


class DiffIterator:

    def __init__(self, diff: str, chunk_size: int):

        self._diff = diff
        self._chunk_size = chunk_size

        length = len(self._diff)
        self._length = length

        self._index = 0

    def __iter__(self):

        return self

    def __next__(self):

        if self._index > self._length:
            raise StopIteration
        else:
            self._index += self._chunk_size
            end = self._index + self._chunk_size
            chunk = self._diff[self._index : end]

            return chunk


class HuggingFaceReviewer:
    task = "text-generation"
    # There are likely more performant models
    # Please see https://www.sbert.net/docs/pretrained_models.html
    embedding_model = "sentence-transformers/all-mpnet-base-v2"
    embedding_task = "feature-extraction"

    def __init__(
        self,
        repo_id: str,
        temperature: float,
        top_p: float,
        top_k: int,
        max_new_tokens: Optional[int] = None,
    ):

        self._repo_id = repo_id
        self._temperature = temperature
        self._top_p = top_p
        self._top_k = top_k
        self._max_new_tokens = max_new_tokens

        self._huggingfacehub_api_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")

    def build_embedding(self):

        cls = self.__class__

        embeddings = HuggingFaceHubEmbeddings(
            model=cls.embedding_model,
            task=cls.embedding_task,
        )
        self._embeddings = embeddings

        return self._embeddings

    def build_endpoint(self):

        cls = self.__class__

        endpoint_kwargs = {
            "repo_id": self._repo_id,
            "task": cls.task,
            "temperature": self._temperature,
            "top_p": self._top_p,
            "top_k": self._top_k,
            "huggingfacehub_api_token": self._huggingfacehub_api_token,
        }

        if self._max_new_tokens is not None:
            endpoint_kwargs["max_new_tokens"] = self._max_new_tokens

        endpoint = HuggingFaceEndpoint(**endpoint_kwargs)
        self._endpoint = endpoint

        return self._endpoint

    def build_tokenizer(self):

        tokenizer = AutoTokenizer.from_pretrained(self._repo_id)
        self._tokenizer = tokenizer

        return self._tokenizer

    def build_model(self):

        self.build_endpoint()
        self.build_tokenizer()

        model = ChatHuggingFace(llm=self._endpoint, tokenizer=self._tokenizer)
        self._model = model

        return self._model

    def generate_summary(self, diff: str):

        # This is the format required for Mistral and other LLMs
        # Please see: https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.2
        prompt = ChatPromptTemplate.from_messages(
            [
                ("user", "Hello"),
                (
                    "assistant",
                    # pylint: disable=line-too-long
                    "Hello, I am a helpful AI software analyst. I assist software developers in reviewing and writing source code for applications.",
                    # pylint: enable=line-too-long
                ),
                (
                    "user",
                    # pylint: disable=line-too-long
                    "Provide a very concise summary for the changes in a git diff generated from a pull request submitted by a developer on GitHub. Importantly, do not explain the 'git diff' command nor reference any git commit hashes in the summary.\n\ngit diff: {diff}",
                    # pylint: enable=line-too-long
                ),
            ],
        )

        chain = prompt | self._model | StrOutputParser()

        inputs = {"diff": diff}

        summary = chain.invoke(inputs)

        return summary

    def generate_review(self, changes: str):

        prompt = ChatPromptTemplate.from_messages(
            [
                ("user", "Hello"),
                (
                    "assistant",
                    # pylint: disable=line-too-long
                    "Hello, I am a helpful AI software analyst. I assist software developers in reviewing and writing source code for applications.",
                    # pylint: enable=line-too-long
                ),
                (
                    "user",
                    # pylint: disable=line-too-long
                    "Summarize the following changes in a git diff generated from a pull request submitted by a developer on GitHub. Importantly, include the line number of the change in the summary. The summary must not exceed 800 characters. Do not specify the total character count for the summary in the answer.\n\ngit diff: {changes}",
                    # pylint: enable=line-too-long
                ),
            ],
        )

        chain = prompt | self._model | StrOutputParser()

        inputs = {"changes": changes}

        review = chain.invoke(inputs)

        return review

    def generate_suggestion(self, changes: str):

        # Prompt for suggested improvements
        prompt = ChatPromptTemplate.from_messages(
            [
                ("user", "Hello"),
                (
                    "assistant",
                    # pylint: disable=line-too-long
                    "Hello, I am a helpful AI software analyst. I assist software developers in reviewing and writing source code for applications.",
                    # pylint: enable=line-too-long
                ),
                (
                    "user",
                    # pylint: disable=line-too-long
                    "Analyze the following changes in a git diff generated from a pull request submitted by a developer on GitHub. If you are able to, determine if these proposed changes might be improved upon in any manner, and recommend any of these improvements. Your answer must not exceed 800 characters.\n\ngit diff: {changes}",
                    # pylint: enable=line-too-long
                ),
            ],
        )

        chain = prompt | self._model | StrOutputParser()

        inputs = {"changes": changes}

        suggestion = chain.invoke(inputs)

        return suggestion

    def invoke(self, diff: str, diff_chunk_size: int):
        """Generate the summary, review, and suggested improvements"""

        # Chunk the prompt
        if diff_chunk_size > 0:
            diff_iter = DiffIterator(diff=diff, chunk_size=diff_chunk_size)
        else:
            diff_iter = [diff]

        self.build_model()

        summary = self.generate_summary(diff=diff)
        summaries = [summary]

        reviews = []
        for changes in diff_iter:

            review = self.generate_review(changes=changes)
            reviews.append(review)

            suggestion = self.generate_suggestion(changes=changes)
            # Suggested improvements are appended directly beneath the overview
            reviews.append(suggestion)

        return summaries, reviews


class ReviewComment:
    delimiter = "\n"
    summaries_header = "## Summary of Proposed Changes"
    reviews_header = "## Suggested Improvements"

    @classmethod
    def format(cls, summaries: List[str], reviews: List[str]) -> str:
        """Format reviews"""

        joined_summaries = cls.delimiter.join(summaries)
        joined_reviews = cls.delimiter.join(reviews)

        # pylint: disable=line-too-long
        comment = f"{cls.summaries_header}{cls.delimiter}{joined_summaries}{cls.delimiter}{cls.reviews_header}{cls.delimiter}{joined_reviews}"
        # pylint: enable=line-too-long

        return comment


@click.command()
@click.option(
    "--diffs",
    type=click.STRING,
    required=True,
    help="Directory path to the file diffs generated for the pull request",
)
@click.option(
    "--diff-chunk-size",
    type=click.INT,
    required=False,
    default=0,
    help="Maximum number of characters for diff chunks for analysis",
)
@click.option(
    "--repo-id",
    type=click.STRING,
    required=False,
    default="mistralai/Mistral-7B-Instruct-v0.2",
    help="HuggingFace model repository ID",
)
@click.option(
    "--temperature", type=click.FLOAT, required=False, default=0.1, help="Temperature"
)
@click.option(
    "--max-new-tokens", type=click.INT, required=False, default=250, help="Max tokens"
)
@click.option("--top-p", type=click.FLOAT, required=False, default=1.0, help="Top N")
@click.option("--top-k", type=click.INT, required=False, default=1.0, help="Top T")
@click.option(
    "--log-level",
    type=click.STRING,
    required=False,
    default="INFO",
    help="Logging level",
)
def main(
    diffs: str,
    diff_chunk_size: int,
    repo_id: str,
    temperature: float,
    max_new_tokens: int,
    top_p: float,
    top_k: int,
    log_level: str,
):

    # Set log level
    logger.level(log_level)

    # Check if necessary environment variables are set or not
    check_required_env_vars()

    for diff_file_name in os.listdir(diffs):

        if not diff_file_name.endswith(".diff"):
            continue

        diff_file_path = os.path.join(diffs, diff_file_name)

        # Open and read the contents from the file generated from `git diff`
        file_diff = ""
        with open(diff_file_path, encoding="utf-8") as fh:

            file_diff = fh.read()

        logger.debug(f"git diff: {file_diff}")

        reviewer_kwargs = {
            "repo_id": repo_id,
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
        }

        if max_new_tokens > 0:
            reviewer_kwargs["max_new_tokens"] = max_new_tokens

        reviewer = HuggingFaceReviewer(**reviewer_kwargs)

        summaries, reviews = reviewer.invoke(
            diff=file_diff, diff_chunk_size=diff_chunk_size
        )

        logger.debug(f"Generated summaries: {summaries}")
        logger.debug(f"Generated reviews: {reviews}")

        # Format reviews
        comment = ReviewComment.format(summaries=summaries, reviews=reviews)

        logger.debug(f"GitHub comment: {comment}")

        github_token = os.getenv("GH_TOKEN")
        repository = os.getenv("GITHUB_REPOSITORY")
        pull_request_number_env = os.getenv("GITHUB_PULL_REQUEST_NUMBER")
        pull_request_number = int(pull_request_number_env)
        git_commit_hash = os.getenv("GIT_COMMIT_HASH")

        pull_request = PullRequest(
            github_token=github_token, repository=repository, number=pull_request_number
        )
        pull_request.comment(commit_hash=git_commit_hash, body=comment)


if __name__ == "__main__":
    # pylint: disable=no-value-for-parameter
    main()
