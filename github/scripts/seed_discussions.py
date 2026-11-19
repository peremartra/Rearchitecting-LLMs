#!/usr/bin/env python3
"""Seed a cohort repository with the book's "Hands-on" discussions.

Copies every discussion of the source repository that carries the `hands-on labs`
label into the repository this script runs in: same title, same body, same labels.
Labels that do not exist yet are created. Discussions whose title already exists
are skipped, so the script can be run more than once.

Requires the GitHub CLI (`gh`) authenticated through the GH_TOKEN environment
variable, and Discussions enabled in the target repository.
"""
import argparse
import json
import os
import re
import subprocess
import sys
import time

DEFAULT_SOURCE = "peremartra/Rearchitecting-LLMs"

SOURCE_Q = """
query($owner: String!, $name: String!, $after: String) {
  repository(owner: $owner, name: $name) {
    discussions(first: 50, after: $after, orderBy: {field: CREATED_AT, direction: ASC}) {
      pageInfo { hasNextPage endCursor }
      nodes {
        title
        body
        labels(first: 20) { nodes { name color description } }
      }
    }
  }
}
"""

TARGET_Q = """
query($owner: String!, $name: String!) {
  repository(owner: $owner, name: $name) {
    id
    hasDiscussionsEnabled
    discussionCategories(first: 25) { nodes { id name } }
  }
}
"""

TITLES_Q = """
query($owner: String!, $name: String!, $after: String) {
  repository(owner: $owner, name: $name) {
    discussions(first: 100, after: $after) {
      pageInfo { hasNextPage endCursor }
      nodes { title }
    }
  }
}
"""

LABELS_Q = """
query($owner: String!, $name: String!, $after: String) {
  repository(owner: $owner, name: $name) {
    labels(first: 100, after: $after) {
      pageInfo { hasNextPage endCursor }
      nodes { id name }
    }
  }
}
"""

CREATE_M = """
mutation($repositoryId: ID!, $categoryId: ID!, $title: String!, $body: String!) {
  createDiscussion(input: {repositoryId: $repositoryId, categoryId: $categoryId, title: $title, body: $body}) {
    discussion { id url }
  }
}
"""

ADD_LABELS_M = """
mutation($labelableId: ID!, $labelIds: [ID!]!) {
  addLabelsToLabelable(input: {labelableId: $labelableId, labelIds: $labelIds}) {
    clientMutationId
  }
}
"""


def gh(args, payload=None):
    result = subprocess.run(
        ["gh", "api", *args],
        input=json.dumps(payload) if payload is not None else None,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        sys.exit("gh api failed: " + " ".join(args) + "\n" + (result.stderr or result.stdout).strip())
    data = json.loads(result.stdout) if result.stdout.strip() else {}
    if isinstance(data, dict) and data.get("errors"):
        sys.exit("GitHub returned errors: " + json.dumps(data["errors"]))
    return data


def graphql(query, **variables):
    return gh(["graphql", "--input", "-"], {"query": query, "variables": variables})["data"]


def paginate(query, connection, **variables):
    """Yield every node of a paginated connection of `repository`."""
    after = None
    while True:
        page = graphql(query, after=after, **variables)["repository"][connection]
        yield from page["nodes"]
        if not page["pageInfo"]["hasNextPage"]:
            return
        after = page["pageInfo"]["endCursor"]


def split_repo(repo):
    owner, _, name = repo.partition("/")
    if not owner or not name:
        sys.exit(f"'{repo}' is not in owner/name form")
    return owner, name


def chapter_key(discussion):
    """Order discussions by chapter label (CH02, CH03...); the rest go last."""
    for label in discussion["labels"]["nodes"]:
        match = re.fullmatch(r"CH(\d+)", label["name"], re.IGNORECASE)
        if match:
            return int(match.group(1))
    return 999


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--source", default=DEFAULT_SOURCE, help="owner/name to copy from")
    parser.add_argument("--target", default=os.environ.get("GITHUB_REPOSITORY"), help="owner/name to copy into")
    parser.add_argument("--label", default="hands-on labs", help="only copy discussions with this label")
    parser.add_argument("--category", default="General", help="category for the new discussions")
    parser.add_argument("--dry-run", action="store_true", help="list what would be created, change nothing")
    args = parser.parse_args()

    if not args.target:
        sys.exit("Set GITHUB_REPOSITORY or pass --target owner/name")
    if args.source.lower() == args.target.lower():
        sys.exit("Source and target are the same repository; nothing to do.")

    s_owner, s_name = split_repo(args.source)
    t_owner, t_name = split_repo(args.target)

    # 1. What to copy.
    wanted = args.label.lower()
    source = [
        d for d in paginate(SOURCE_Q, "discussions", owner=s_owner, name=s_name)
        if any(label["name"].lower() == wanted for label in d["labels"]["nodes"])
    ]
    if not source:
        sys.exit(f"No discussions labeled '{args.label}' found in {args.source}.")
    source.sort(key=chapter_key)

    # 2. Where to copy it.
    target = graphql(TARGET_Q, owner=t_owner, name=t_name)["repository"]
    if not target["hasDiscussionsEnabled"]:
        sys.exit(
            f"Discussions are not enabled in {args.target}. Enable them in "
            "Settings > General > Features > Discussions and run this again."
        )
    categories = {c["name"].lower(): c["id"] for c in target["discussionCategories"]["nodes"]}
    if args.category.lower() not in categories:
        available = sorted(c["name"] for c in target["discussionCategories"]["nodes"])
        sys.exit(f"Category '{args.category}' not found. Available: {', '.join(available)}")
    category_id = categories[args.category.lower()]

    existing_titles = {d["title"] for d in paginate(TITLES_Q, "discussions", owner=t_owner, name=t_name)}
    label_ids = {l["name"].lower(): l["id"] for l in paginate(LABELS_Q, "labels", owner=t_owner, name=t_name)}

    # 3. Labels.
    needed = {}
    for d in source:
        for label in d["labels"]["nodes"]:
            needed.setdefault(label["name"].lower(), label)
    missing = [label for key, label in needed.items() if key not in label_ids]
    for label in missing:
        print(f"label  + {label['name']}")
        if not args.dry_run:
            gh(
                ["--method", "POST", f"repos/{args.target}/labels", "--input", "-"],
                {"name": label["name"], "color": label["color"], "description": label.get("description") or ""},
            )
    if missing and not args.dry_run:
        label_ids = {l["name"].lower(): l["id"] for l in paginate(LABELS_Q, "labels", owner=t_owner, name=t_name)}

    # 4. Discussions.
    created = skipped = 0
    for d in source:
        if d["title"] in existing_titles:
            print(f"skip   = {d['title']}")
            skipped += 1
            continue
        print(f"create + {d['title']}")
        created += 1
        if args.dry_run:
            continue
        new = graphql(
            CREATE_M, repositoryId=target["id"], categoryId=category_id, title=d["title"], body=d["body"]
        )["createDiscussion"]["discussion"]
        ids = [label_ids[l["name"].lower()] for l in d["labels"]["nodes"]]
        if ids:
            graphql(ADD_LABELS_M, labelableId=new["id"], labelIds=ids)
        time.sleep(1)  # stay well under GitHub's content-creation rate limits

    verb = "would create" if args.dry_run else "created"
    print(f"\nDone: {verb} {created} discussion(s), {len(missing)} label(s); skipped {skipped} existing.")


if __name__ == "__main__":
    main()
