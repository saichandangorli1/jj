// Copyright 2022 The Jujutsu Authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

use testutils::git;

use crate::common::CommandOutput;
use crate::common::TestEnvironment;
use crate::common::TestWorkDir;

fn git_repo_dir_for_jj_repo(work_dir: &TestWorkDir<'_>) -> std::path::PathBuf {
    work_dir
        .root()
        .join(".jj")
        .join("repo")
        .join("store")
        .join("git")
}

fn set_up(test_env: &TestEnvironment) {
    test_env.run_jj_in(".", ["git", "init", "origin"]).success();
    let origin_dir = test_env.work_dir("origin");
    let origin_git_repo_path = git_repo_dir_for_jj_repo(&origin_dir);

    origin_dir
        .run_jj(["describe", "-m=description 1"])
        .success();
    origin_dir
        .run_jj(["bookmark", "create", "-r@", "bookmark1"])
        .success();
    origin_dir
        .run_jj(["new", "root()", "-m=description 2"])
        .success();
    origin_dir
        .run_jj(["bookmark", "create", "-r@", "bookmark2"])
        .success();
    origin_dir.run_jj(["git", "export"]).success();

    test_env
        .run_jj_in(
            ".",
            [
                "git",
                "clone",
                "--config=git.auto-local-bookmark=true",
                origin_git_repo_path.to_str().unwrap(),
                "local",
            ],
        )
        .success();
}

#[test]
fn test_git_push_nothing() {
    let test_env = TestEnvironment::default();
    set_up(&test_env);
    let work_dir = test_env.work_dir("local");
    // Show the setup. `insta` has trouble if this is done inside `set_up()`
    insta::assert_snapshot!(get_bookmark_output(&work_dir), @r"
    bookmark1: qpvuntsm 9b2e76de (empty) description 1
      @origin: qpvuntsm 9b2e76de (empty) description 1
    bookmark2: zsuskuln 38a20473 (empty) description 2
      @origin: zsuskuln 38a20473 (empty) description 2
    [EOF]
    ");
    // No bookmarks to push yet
    let output = work_dir.run_jj(["git", "push", "--all"]);
    insta::assert_snapshot!(output, @r"
    ------- stderr -------
    Nothing changed.
    [EOF]
    ");
}

#[test]
fn test_git_push_current_bookmark() {
    let test_env = TestEnvironment::default();
    set_up(&test_env);
    let work_dir = test_env.work_dir("local");
    test_env.add_config(r#"revset-aliases."immutable_heads()" = "none()""#);
    // Update some bookmarks. `bookmark1` is not a current bookmark, but
    // `bookmark2` and `my-bookmark` are.
    work_dir
        .run_jj(["describe", "bookmark1", "-m", "modified bookmark1 commit"])
        .success();
    work_dir.run_jj(["new", "bookmark2"]).success();
    work_dir
        .run_jj(["bookmark", "set", "bookmark2", "-r@"])
        .success();
    work_dir
        .run_jj(["bookmark", "create", "-r@", "my-bookmark"])
        .success();
    work_dir.run_jj(["describe", "-m", "foo"]).success();
    // Check the setup
    insta::assert_snapshot!(get_bookmark_output(&work_dir), @r"
    bookmark1: qpvuntsm e5ce6d9a (empty) modified bookmark1 commit
      @origin (ahead by 1 commits, behind by 1 commits): qpvuntsm hidden 9b2e76de (empty) description 1
    bookmark2: yostqsxw 88ca14a7 (empty) foo
      @origin (behind by 1 commits): zsuskuln 38a20473 (empty) description 2
    my-bookmark: yostqsxw 88ca14a7 (empty) foo
    [EOF]
    ");
    // First dry-run. `bookmark1` should not get pushed.
    let output = work_dir.run_jj(["git", "push", "--allow-new", "--dry-run"]);
    insta::assert_snapshot!(output, @r"
    ------- stderr -------
    Changes to push to origin:
      Move forward bookmark bookmark2 from 38a204733702 to 88ca14a7d46f
      Add bookmark my-bookmark to 88ca14a7d46f
    Dry-run requested, not pushing.
    [EOF]
    ");
    let output = work_dir.run_jj(["git", "push", "--allow-new"]);
    insta::assert_snapshot!(output, @r"
    ------- stderr -------
    Changes to push to origin:
      Move forward bookmark bookmark2 from 38a204733702 to 88ca14a7d46f
      Add bookmark my-bookmark to 88ca14a7d46f
    [EOF]
    ");
    insta::assert_snapshot!(get_bookmark_output(&work_dir), @r"
    bookmark1: qpvuntsm e5ce6d9a (empty) modified bookmark1 commit
      @origin (ahead by 1 commits, behind by 1 commits): qpvuntsm hidden 9b2e76de (empty) description 1
    bookmark2: yostqsxw 88ca14a7 (empty) foo
      @origin: yostqsxw 88ca14a7 (empty) foo
    my-bookmark: yostqsxw 88ca14a7 (empty) foo
      @origin: yostqsxw 88ca14a7 (empty) foo
    [EOF]
    ");

    // Try pushing backwards
    work_dir
        .run_jj([
            "bookmark",
            "set",
            "bookmark2",
            "-rbookmark2-",
            "--allow-backwards",
        ])
        .success();
    // This behavior is a strangeness of our definition of the default push revset.
    // We could consider changing it.
    let output = work_dir.run_jj(["git", "push"]);
    insta::assert_snapshot!(output, @r"
    ------- stderr -------
    Warning: No bookmarks found in the default push revset: remote_bookmarks(remote=remote)..@
    Nothing changed.
    [EOF]
    ");
    // We can move a bookmark backwards
    let output = work_dir.run_jj(["git", "push", "-bbookmark2"]);
    insta::assert_snapshot!(output, @r"
    ------- stderr -------
    Changes to push to origin:
      Move backward bookmark bookmark2 from 88ca14a7d46f to 38a204733702
    [EOF]
    ");
}

#[test]
fn test_git_push_parent_bookmark() {
    let test_env = TestEnvironment::default();
    set_up(&test_env);
    let work_dir = test_env.work_dir("local");
    test_env.add_config(r#"revset-aliases."immutable_heads()" = "none()""#);
    work_dir.run_jj(["edit", "bookmark1"]).success();
    work_dir
        .run_jj(["describe", "-m", "modified bookmark1 commit"])
        .success();
    work_dir
        .run_jj(["new", "-m", "non-empty description"])
        .success();
    work_dir.write_file("file", "file");
    let output = work_dir.run_jj(["git", "push"]);
    insta::assert_snapshot!(output, @r"
    ------- stderr -------
    Changes to push to origin:
      Move sideways bookmark bookmark1 from 9b2e76de3920 to 80560a3e08e2
    [EOF]
    ");
}

#[test]
fn test_git_push_no_matching_bookmark() {
    let test_env = TestEnvironment::default();
    set_up(&test_env);
    let work_dir = test_env.work_dir("local");
    work_dir.run_jj(["new"]).success();
    let output = work_dir.run_jj(["git", "push"]);
    insta::assert_snapshot!(output, @r"
    ------- stderr -------
    Nothing changed.
    [EOF]
    ");
}

#[test]
fn test_git_push_matching_bookmark_unchanged() {
    let test_env = TestEnvironment::default();
    set_up(&test_env);
    let work_dir = test_env.work_dir("local");
    work_dir.run_jj(["new", "bookmark1"]).success();
    let output = work_dir.run_jj(["git", "push"]);
    insta::assert_snapshot!(output, @r"
    ------- stderr -------
    Nothing changed.
    [EOF]
    ");
}

/// Test that `jj git push` without arguments pushes a bookmark to the specified
/// remote even if it's already up to date on another remote
/// (`remote_bookmarks(remote=<remote>)..@` vs. `remote_bookmarks()..@`).
#[test]
fn test_git_push_other_remote_has_bookmark() {
    let test_env = TestEnvironment::default();
    set_up(&test_env);
    let work_dir = test_env.work_dir("local");
    test_env.add_config(r#"revset-aliases."immutable_heads()" = "none()""#);
    // Create another remote (but actually the same)
    let other_remote_path = test_env
        .env_root()
        .join("origin")
        .join(".jj")
        .join("repo")
        .join("store")
        .join("git");
    work_dir
        .run_jj([
            "git",
            "remote",
            "add",
            "other",
            other_remote_path.to_str().unwrap(),
        ])
        .success();
    // Modify bookmark1 and push it to `origin`
    work_dir.run_jj(["edit", "bookmark1"]).success();
    work_dir.run_jj(["describe", "-m=modified"]).success();
    let output = work_dir.run_jj(["git", "push"]);
    insta::assert_snapshot!(output, @r"
    ------- stderr -------
    Changes to push to origin:
      Move sideways bookmark bookmark1 from 9b2e76de3920 to a843bfad2abb
    [EOF]
    ");
    // Since it's already pushed to origin, nothing will happen if push again
    let output = work_dir.run_jj(["git", "push"]);
    insta::assert_snapshot!(output, @r"
    ------- stderr -------
    Warning: No bookmarks found in the default push revset: remote_bookmarks(remote=remote)..@
    Nothing changed.
    [EOF]
    ");
    // The bookmark was moved on the "other" remote as well (since it's actually the
    // same remote), but `jj` is not aware of that since it thinks this is a
    // different remote. So, the push should fail.
    //
    // But it succeeds! That's because the bookmark is created at the same location
    // as it is on the remote. This would also work for a descendant.
    //
    // TODO: Saner test?
    let output = work_dir.run_jj(["git", "push", "--allow-new", "--remote=other"]);
    insta::assert_snapshot!(output, @r"
    ------- stderr -------
    Changes to push to other:
      Add bookmark bookmark1 to a843bfad2abb
    [EOF]
    ");
}

#[test]
fn test_git_push_forward_unexpectedly_moved() {
    let test_env = TestEnvironment::default();
    set_up(&test_env);
    let work_dir = test_env.work_dir("local");

    // Move bookmark1 forward on the remote
    let origin_dir = test_env.work_dir("origin");
    origin_dir
        .run_jj(["new", "bookmark1", "-m=remote"])
        .success();
    origin_dir.write_file("remote", "remote");
    origin_dir
        .run_jj(["bookmark", "set", "bookmark1", "-r@"])
        .success();
    origin_dir.run_jj(["git", "export"]).success();

    // Move bookmark1 forward to another commit locally
    work_dir.run_jj(["new", "bookmark1", "-m=local"]).success();
    work_dir.write_file("local", "local");
    work_dir
        .run_jj(["bookmark", "set", "bookmark1", "-r@"])
        .success();

    // Pushing should fail
    let output = work_dir.run_jj(["git", "push"]);
    insta::assert_snapshot!(output, @r"
    ------- stderr -------
    Changes to push to origin:
      Move forward bookmark bookmark1 from 9b2e76de3920 to 624f94a35f00
    Error: Failed to push some bookmarks
    Hint: The following references unexpectedly moved on the remote:
      refs/heads/bookmark1 (reason: stale info)
    Hint: Try fetching from the remote, then make the bookmark point to where you want it to be, and push again.
    [EOF]
    [exit status: 1]
    ");
}

#[test]
fn test_git_push_sideways_unexpectedly_moved() {
    let test_env = TestEnvironment::default();
    set_up(&test_env);
    let work_dir = test_env.work_dir("local");

    // Move bookmark1 forward on the remote
    let origin_dir = test_env.work_dir("origin");
    origin_dir
        .run_jj(["new", "bookmark1", "-m=remote"])
        .success();
    origin_dir.write_file("remote", "remote");
    origin_dir
        .run_jj(["bookmark", "set", "bookmark1", "-r@"])
        .success();
    insta::assert_snapshot!(get_bookmark_output(&origin_dir), @r"
    bookmark1: vruxwmqv 7ce4029e remote
      @git (behind by 1 commits): qpvuntsm 9b2e76de (empty) description 1
    bookmark2: zsuskuln 38a20473 (empty) description 2
      @git: zsuskuln 38a20473 (empty) description 2
    [EOF]
    ");
    origin_dir.run_jj(["git", "export"]).success();

    // Move bookmark1 sideways to another commit locally
    work_dir.run_jj(["new", "root()", "-m=local"]).success();
    work_dir.write_file("local", "local");
    work_dir
        .run_jj(["bookmark", "set", "bookmark1", "--allow-backwards", "-r@"])
        .success();
    insta::assert_snapshot!(get_bookmark_output(&work_dir), @r"
    bookmark1: kmkuslsw 827b8a38 local
      @origin (ahead by 1 commits, behind by 1 commits): qpvuntsm 9b2e76de (empty) description 1
    bookmark2: zsuskuln 38a20473 (empty) description 2
      @origin: zsuskuln 38a20473 (empty) description 2
    [EOF]
    ");

    let output = work_dir.run_jj(["git", "push"]);
    insta::assert_snapshot!(output, @r"
    ------- stderr -------
    Changes to push to origin:
      Move sideways bookmark bookmark1 from 9b2e76de3920 to 827b8a385853
    Error: Failed to push some bookmarks
    Hint: The following references unexpectedly moved on the remote:
      refs/heads/bookmark1 (reason: stale info)
    Hint: Try fetching from the remote, then make the bookmark point to where you want it to be, and push again.
    [EOF]
    [exit status: 1]
    ");
}

// This tests whether the push checks that the remote bookmarks are in expected
// positions.
#[test]
fn test_git_push_deletion_unexpectedly_moved() {
    let test_env = TestEnvironment::default();
    set_up(&test_env);
    let work_dir = test_env.work_dir("local");

    // Move bookmark1 forward on the remote
    let origin_dir = test_env.work_dir("origin");
    origin_dir
        .run_jj(["new", "bookmark1", "-m=remote"])
        .success();
    origin_dir.write_file("remote", "remote");
    origin_dir
        .run_jj(["bookmark", "set", "bookmark1", "-r@"])
        .success();
    insta::assert_snapshot!(get_bookmark_output(&origin_dir), @r"
    bookmark1: vruxwmqv 7ce4029e remote
      @git (behind by 1 commits): qpvuntsm 9b2e76de (empty) description 1
    bookmark2: zsuskuln 38a20473 (empty) description 2
      @git: zsuskuln 38a20473 (empty) description 2
    [EOF]
    ");
    origin_dir.run_jj(["git", "export"]).success();

    // Delete bookmark1 locally
    work_dir
        .run_jj(["bookmark", "delete", "bookmark1"])
        .success();
    insta::assert_snapshot!(get_bookmark_output(&work_dir), @r"
    bookmark1 (deleted)
      @origin: qpvuntsm 9b2e76de (empty) description 1
    bookmark2: zsuskuln 38a20473 (empty) description 2
      @origin: zsuskuln 38a20473 (empty) description 2
    [EOF]
    ");

    let output = work_dir.run_jj(["git", "push", "--bookmark", "bookmark1"]);
    insta::assert_snapshot!(output, @r"
    ------- stderr -------
    Changes to push to origin:
      Delete bookmark bookmark1 from 9b2e76de3920
    Error: Failed to push some bookmarks
    Hint: The following references unexpectedly moved on the remote:
      refs/heads/bookmark1 (reason: stale info)
    Hint: Try fetching from the remote, then make the bookmark point to where you want it to be, and push again.
    [EOF]
    [exit status: 1]
    ");
}

#[test]
fn test_git_push_unexpectedly_deleted() {
    let test_env = TestEnvironment::default();
    set_up(&test_env);
    let work_dir = test_env.work_dir("local");

    // Delete bookmark1 forward on the remote
    let origin_dir = test_env.work_dir("origin");
    origin_dir
        .run_jj(["bookmark", "delete", "bookmark1"])
        .success();
    insta::assert_snapshot!(get_bookmark_output(&origin_dir), @r"
    bookmark1 (deleted)
      @git: qpvuntsm 9b2e76de (empty) description 1
    bookmark2: zsuskuln 38a20473 (empty) description 2
      @git: zsuskuln 38a20473 (empty) description 2
    [EOF]
    ");
    origin_dir.run_jj(["git", "export"]).success();

    // Move bookmark1 sideways to another commit locally
    work_dir.run_jj(["new", "root()", "-m=local"]).success();
    work_dir.write_file("local", "local");
    work_dir
        .run_jj(["bookmark", "set", "bookmark1", "--allow-backwards", "-r@"])
        .success();
    insta::assert_snapshot!(get_bookmark_output(&work_dir), @r"
    bookmark1: kpqxywon 09919fb0 local
      @origin (ahead by 1 commits, behind by 1 commits): qpvuntsm 9b2e76de (empty) description 1
    bookmark2: zsuskuln 38a20473 (empty) description 2
      @origin: zsuskuln 38a20473 (empty) description 2
    [EOF]
    ");

    // Pushing a moved bookmark fails if deleted on remote
    let output = work_dir.run_jj(["git", "push"]);
    insta::assert_snapshot!(output, @r"
    ------- stderr -------
    Changes to push to origin:
      Move sideways bookmark bookmark1 from 9b2e76de3920 to 09919fb051bf
    Error: Failed to push some bookmarks
    Hint: The following references unexpectedly moved on the remote:
      refs/heads/bookmark1 (reason: stale info)
    Hint: Try fetching from the remote, then make the bookmark point to where you want it to be, and push again.
    [EOF]
    [exit status: 1]
    ");

    work_dir
        .run_jj(["bookmark", "delete", "bookmark1"])
        .success();
    insta::assert_snapshot!(get_bookmark_output(&work_dir), @r"
    bookmark1 (deleted)
      @origin: qpvuntsm 9b2e76de (empty) description 1
    bookmark2: zsuskuln 38a20473 (empty) description 2
      @origin: zsuskuln 38a20473 (empty) description 2
    [EOF]
    ");

    // git does not allow to push a deleted bookmark if we expect it to exist even
    // though it was already deleted
    let output = work_dir.run_jj(["git", "push", "-bbookmark1"]);
    insta::assert_snapshot!(output, @r"
    ------- stderr -------
    Changes to push to origin:
      Delete bookmark bookmark1 from 9b2e76de3920
    Error: Failed to push some bookmarks
    Hint: The following references unexpectedly moved on the remote:
      refs/heads/bookmark1 (reason: stale info)
    Hint: Try fetching from the remote, then make the bookmark point to where you want it to be, and push again.
    [EOF]
    [exit status: 1]
    ");
}

#[test]
fn test_git_push_creation_unexpectedly_already_exists() {
    let test_env = TestEnvironment::default();
    set_up(&test_env);
    let work_dir = test_env.work_dir("local");

    // Forget bookmark1 locally
    work_dir
        .run_jj(["bookmark", "forget", "--include-remotes", "bookmark1"])
        .success();

    // Create a new branh1
    work_dir
        .run_jj(["new", "root()", "-m=new bookmark1"])
        .success();
    work_dir.write_file("local", "local");
    work_dir
        .run_jj(["bookmark", "create", "-r@", "bookmark1"])
        .success();
    insta::assert_snapshot!(get_bookmark_output(&work_dir), @r"
    bookmark1: yostqsxw a43cb801 new bookmark1
    bookmark2: zsuskuln 38a20473 (empty) description 2
      @origin: zsuskuln 38a20473 (empty) description 2
    [EOF]
    ");

    let output = work_dir.run_jj(["git", "push", "--allow-new"]);
    insta::assert_snapshot!(output, @r"
    ------- stderr -------
    Changes to push to origin:
      Add bookmark bookmark1 to a43cb8011c85
    Error: Failed to push some bookmarks
    Hint: The following references unexpectedly moved on the remote:
      refs/heads/bookmark1 (reason: stale info)
    Hint: Try fetching from the remote, then make the bookmark point to where you want it to be, and push again.
    [EOF]
    [exit status: 1]
    ");
}

#[test]
fn test_git_push_locally_created_and_rewritten() {
    let test_env = TestEnvironment::default();
    set_up(&test_env);
    let work_dir = test_env.work_dir("local");
    // Ensure that remote bookmarks aren't tracked automatically
    test_env.add_config("git.auto-local-bookmark = false");

    // Push locally-created bookmark
    work_dir.run_jj(["new", "root()", "-mlocal 1"]).success();
    work_dir
        .run_jj(["bookmark", "create", "-r@", "my"])
        .success();
    let output = work_dir.run_jj(["git", "push"]);
    insta::assert_snapshot!(output, @r"
    ------- stderr -------
    Warning: Refusing to create new remote bookmark my@origin
    Hint: Use --allow-new to push new bookmark. Use --remote to specify the remote to push to.
    Nothing changed.
    [EOF]
    ");
    // Either --allow-new or git.push-new-bookmarks=true should work
    let output = work_dir.run_jj(["git", "push", "--allow-new", "--dry-run"]);
    insta::assert_snapshot!(output, @r"
    ------- stderr -------
    Changes to push to origin:
      Add bookmark my to e0cba5e497ee
    Dry-run requested, not pushing.
    [EOF]
    ");
    let output = work_dir.run_jj(["git", "push", "--config=git.push-new-bookmarks=true"]);
    insta::assert_snapshot!(output, @r"
    ------- stderr -------
    Changes to push to origin:
      Add bookmark my to e0cba5e497ee
    [EOF]
    ");

    // Rewrite it and push again, which would fail if the pushed bookmark weren't
    // set to "tracking"
    work_dir.run_jj(["describe", "-mlocal 2"]).success();
    insta::assert_snapshot!(get_bookmark_output(&work_dir), @r"
    bookmark1: qpvuntsm 9b2e76de (empty) description 1
      @origin: qpvuntsm 9b2e76de (empty) description 1
    bookmark2: zsuskuln 38a20473 (empty) description 2
      @origin: zsuskuln 38a20473 (empty) description 2
    my: vruxwmqv 5eb416c1 (empty) local 2
      @origin (ahead by 1 commits, behind by 1 commits): vruxwmqv hidden e0cba5e4 (empty) local 1
    [EOF]
    ");
    let output = work_dir.run_jj(["git", "push"]);
    insta::assert_snapshot!(output, @r"
    ------- stderr -------
    Changes to push to origin:
      Move sideways bookmark my from e0cba5e497ee to 5eb416c1ff97
    [EOF]
    ");
}

#[test]
fn test_git_push_multiple() {
    let test_env = TestEnvironment::default();
    set_up(&test_env);
    let work_dir = test_env.work_dir("local");
    work_dir
        .run_jj(["bookmark", "delete", "bookmark1"])
        .success();
    work_dir
        .run_jj(["bookmark", "set", "--allow-backwards", "bookmark2", "-r@"])
        .success();
    work_dir
        .run_jj(["bookmark", "create", "-r@", "my-bookmark"])
        .success();
    work_dir.run_jj(["describe", "-m", "foo"]).success();
    // Check the setup
    insta::assert_snapshot!(get_bookmark_output(&work_dir), @r"
    bookmark1 (deleted)
      @origin: qpvuntsm 9b2e76de (empty) description 1
    bookmark2: yqosqzyt 352fa187 (empty) foo
      @origin (ahead by 1 commits, behind by 1 commits): zsuskuln 38a20473 (empty) description 2
    my-bookmark: yqosqzyt 352fa187 (empty) foo
    [EOF]
    ");
    // First dry-run
    let output = work_dir.run_jj(["git", "push", "--all", "--deleted", "--dry-run"]);
    insta::assert_snapshot!(output, @r"
    ------- stderr -------
    Changes to push to origin:
      Delete bookmark bookmark1 from 9b2e76de3920
      Move sideways bookmark bookmark2 from 38a204733702 to 352fa1879f75
      Add bookmark my-bookmark to 352fa1879f75
    Dry-run requested, not pushing.
    [EOF]
    ");
    // Dry run requesting two specific bookmarks
    let output = work_dir.run_jj([
        "git",
        "push",
        "--allow-new",
        "-b=bookmark1",
        "-b=my-bookmark",
        "--dry-run",
    ]);
    insta::assert_snapshot!(output, @r"
    ------- stderr -------
    Changes to push to origin:
      Delete bookmark bookmark1 from 9b2e76de3920
      Add bookmark my-bookmark to 352fa1879f75
    Dry-run requested, not pushing.
    [EOF]
    ");
    // Dry run requesting two specific bookmarks twice
    let output = work_dir.run_jj([
        "git",
        "push",
        "--allow-new",
        "-b=bookmark1",
        "-b=my-bookmark",
        "-b=bookmark1",
        "-b=glob:my-*",
        "--dry-run",
    ]);
    insta::assert_snapshot!(output, @r"
    ------- stderr -------
    Changes to push to origin:
      Delete bookmark bookmark1 from 9b2e76de3920
      Add bookmark my-bookmark to 352fa1879f75
    Dry-run requested, not pushing.
    [EOF]
    ");
    // Dry run with glob pattern
    let output = work_dir.run_jj(["git", "push", "-b=glob:bookmark?", "--dry-run"]);
    insta::assert_snapshot!(output, @r"
    ------- stderr -------
    Changes to push to origin:
      Delete bookmark bookmark1 from 9b2e76de3920
      Move sideways bookmark bookmark2 from 38a204733702 to 352fa1879f75
    Dry-run requested, not pushing.
    [EOF]
    ");

    // Unmatched bookmark name is error
    let output = work_dir.run_jj(["git", "push", "-b=foo"]);
    insta::assert_snapshot!(output, @r"
    ------- stderr -------
    Error: No such bookmark: foo
    [EOF]
    [exit status: 1]
    ");
    let output = work_dir.run_jj(["git", "push", "-b=foo", "-b=glob:?bookmark"]);
    insta::assert_snapshot!(output, @r"
    ------- stderr -------
    Error: No matching bookmarks for patterns: foo, ?bookmark
    [EOF]
    [exit status: 1]
    ");

    // --deleted is required to push deleted bookmarks even with --all
    let output = work_dir.run_jj(["git", "push", "--all", "--dry-run"]);
    insta::assert_snapshot!(output, @r"
    ------- stderr -------
    Warning: Refusing to push deleted bookmark bookmark1
    Hint: Push deleted bookmarks with --deleted or forget the bookmark to suppress this warning.
    Changes to push to origin:
      Move sideways bookmark bookmark2 from 38a204733702 to 352fa1879f75
      Add bookmark my-bookmark to 352fa1879f75
    Dry-run requested, not pushing.
    [EOF]
    ");
    let output = work_dir.run_jj(["git", "push", "--all", "--deleted", "--dry-run"]);
    insta::assert_snapshot!(output, @r"
    ------- stderr -------
    Changes to push to origin:
      Delete bookmark bookmark1 from 9b2e76de3920
      Move sideways bookmark bookmark2 from 38a204733702 to 352fa1879f75
      Add bookmark my-bookmark to 352fa1879f75
    Dry-run requested, not pushing.
    [EOF]
    ");

    let output = work_dir.run_jj(["git", "push", "--all", "--deleted"]);
    insta::assert_snapshot!(output, @r"
    ------- stderr -------
    Changes to push to origin:
      Delete bookmark bookmark1 from 9b2e76de3920
      Move sideways bookmark bookmark2 from 38a204733702 to 352fa1879f75
      Add bookmark my-bookmark to 352fa1879f75
    [EOF]
    ");
    insta::assert_snapshot!(get_bookmark_output(&work_dir), @r"
    bookmark2: yqosqzyt 352fa187 (empty) foo
      @origin: yqosqzyt 352fa187 (empty) foo
    my-bookmark: yqosqzyt 352fa187 (empty) foo
      @origin: yqosqzyt 352fa187 (empty) foo
    [EOF]
    ");
    let output = work_dir.run_jj(["log", "-rall()"]);
    insta::assert_snapshot!(output, @r"
    @  yqosqzyt test.user@example.com 2001-02-03 08:05:17 bookmark2 my-bookmark 352fa187
    │  (empty) foo
    │ ○  zsuskuln test.user@example.com 2001-02-03 08:05:10 38a20473
    ├─╯  (empty) description 2
    │ ○  qpvuntsm test.user@example.com 2001-02-03 08:05:08 9b2e76de
    ├─╯  (empty) description 1
    ◆  zzzzzzzz root() 00000000
    [EOF]
    ");
}

#[test]
fn test_git_push_changes() {
    let test_env = TestEnvironment::default();
    set_up(&test_env);
    let work_dir = test_env.work_dir("local");
    work_dir.run_jj(["describe", "-m", "foo"]).success();
    work_dir.write_file("file", "contents");
    work_dir.run_jj(["new", "-m", "bar"]).success();
    work_dir.write_file("file", "modified");

    let output = work_dir.run_jj(["git", "push", "--change", "@"]);
    insta::assert_snapshot!(output, @r"
    ------- stderr -------
    Creating bookmark push-yostqsxwqrlt for revision yostqsxwqrlt
    Changes to push to origin:
      Add bookmark push-yostqsxwqrlt to 916414184c47
    [EOF]
    ");
    // test pushing two changes at once
    work_dir.write_file("file", "modified2");
    let output = work_dir.run_jj(["git", "push", "-c=(@|@-)"]);
    insta::assert_snapshot!(output, @r"
    ------- stderr -------
    Error: Revset `(@|@-)` resolved to more than one revision
    Hint: The revset `(@|@-)` resolved to these revisions:
      yostqsxw 2723f611 push-yostqsxwqrlt* | bar
      yqosqzyt 0f8164cd foo
    Hint: Prefix the expression with `all:` to allow any number of revisions (i.e. `all:(@|@-)`).
    [EOF]
    [exit status: 1]
    ");
    // test pushing two changes at once, part 2
    let output = work_dir.run_jj(["git", "push", "-c=all:(@|@-)"]);
    insta::assert_snapshot!(output, @r"
    ------- stderr -------
    Creating bookmark push-yqosqzytrlsw for revision yqosqzytrlsw
    Changes to push to origin:
      Move sideways bookmark push-yostqsxwqrlt from 916414184c47 to 2723f6111cb9
      Add bookmark push-yqosqzytrlsw to 0f8164cd580b
    [EOF]
    ");
    // specifying the same change twice doesn't break things
    work_dir.write_file("file", "modified3");
    let output = work_dir.run_jj(["git", "push", "-c=all:(@|@)"]);
    insta::assert_snapshot!(output, @r"
    ------- stderr -------
    Changes to push to origin:
      Move sideways bookmark push-yostqsxwqrlt from 2723f6111cb9 to 7436a8a600a4
    [EOF]
    ");

    // specifying the same bookmark with --change/--bookmark doesn't break things
    work_dir.write_file("file", "modified4");
    let output = work_dir.run_jj(["git", "push", "-c=@", "-b=push-yostqsxwqrlt"]);
    insta::assert_snapshot!(output, @r"
    ------- stderr -------
    Changes to push to origin:
      Move sideways bookmark push-yostqsxwqrlt from 7436a8a600a4 to a8b93bdd0f68
    [EOF]
    ");

    // try again with --change that could move the bookmark forward
    work_dir.write_file("file", "modified5");
    work_dir
        .run_jj([
            "bookmark",
            "set",
            "-r=@-",
            "--allow-backwards",
            "push-yostqsxwqrlt",
        ])
        .success();
    let output = work_dir.run_jj(["status"]);
    insta::assert_snapshot!(output, @r"
    Working copy changes:
    M file
    Working copy  (@) : yostqsxw 4b18f5ea bar
    Parent commit (@-): yqosqzyt 0f8164cd push-yostqsxwqrlt* push-yqosqzytrlsw | foo
    [EOF]
    ");
    let output = work_dir.run_jj(["git", "push", "-c=@", "-b=push-yostqsxwqrlt"]);
    insta::assert_snapshot!(output, @r"
    ------- stderr -------
    Error: Bookmark already exists: push-yostqsxwqrlt
    Hint: Use 'jj bookmark move' to move it, and 'jj git push -b push-yostqsxwqrlt [--allow-new]' to push it
    [EOF]
    [exit status: 1]
    ");
    let output = work_dir.run_jj(["status"]);
    insta::assert_snapshot!(output, @r"
    Working copy changes:
    M file
    Working copy  (@) : yostqsxw 4b18f5ea bar
    Parent commit (@-): yqosqzyt 0f8164cd push-yostqsxwqrlt* push-yqosqzytrlsw | foo
    [EOF]
    ");

    // Test changing `git.push-bookmark-prefix`. It causes us to push again.
    let output = work_dir.run_jj([
        "git",
        "push",
        "--config=git.push-bookmark-prefix=test-",
        "--change=@",
    ]);
    insta::assert_snapshot!(output, @r"
    ------- stderr -------
    Creating bookmark test-yostqsxwqrlt for revision yostqsxwqrlt
    Changes to push to origin:
      Add bookmark test-yostqsxwqrlt to 4b18f5ea2994
    [EOF]
    ");
}

#[test]
fn test_git_push_changes_with_name() {
    let test_env = TestEnvironment::default();
    set_up(&test_env);
    let work_dir = test_env.work_dir("local");
    work_dir.run_jj(["describe", "-m", "foo"]).success();
    work_dir.write_file("file", "contents");
    work_dir.run_jj(["new", "-m", "pushed"]).success();
    work_dir.write_file("file", "modified");

    // Normal behavior.
    let output = work_dir.run_jj(["git", "push", "--named", "b1=@"]);
    insta::assert_snapshot!(output, @r"
    ------- stderr -------
    Changes to push to origin:
      Add bookmark b1 to 5f4f9a466c96
    [EOF]
    ");
    // Spaces before the = sign are treated like part of the bookmark name and such
    // bookmarks cannot be pushed.
    let output = work_dir.run_jj(["git", "push", "--named", "b1 = @"]);
    insta::assert_snapshot!(output, @r"
    ------- stderr -------
    Error: Could not parse 'b1 ' as a bookmark name
    Caused by:
    1: Failed to parse bookmark name: Syntax error
    2:  --> 1:3
      |
    1 | b1 
      |   ^---
      |
      = expected <EOI>
    Hint: For example, `--named myfeature=@` is valid syntax
    [EOF]
    [exit status: 2]
    ");
    // test pushing a change with an empty name
    let output = work_dir.run_jj(["git", "push", "--named", "=@"]);
    insta::assert_snapshot!(output, @r"
    ------- stderr -------
    Error: Argument '=@' must have the form NAME=REVISION, with both NAME and REVISION non-empty
    Hint: For example, `--named myfeature=@` is valid syntax
    [EOF]
    [exit status: 2]
    ");
    // Unparsable name
    let output = work_dir.run_jj(["git", "push", "--named", ":!:=@"]);
    insta::assert_snapshot!(output, @r"
    ------- stderr -------
    Error: Could not parse ':!:' as a bookmark name
    Caused by:
    1: Failed to parse bookmark name: Syntax error
    2:  --> 1:1
      |
    1 | :!:
      | ^---
      |
      = expected <identifier>, <string_literal>, or <raw_string_literal>
    Hint: For example, `--named myfeature=@` is valid syntax
    [EOF]
    [exit status: 2]
    ");
    // test pushing a change with an empty revision
    let output = work_dir.run_jj(["git", "push", "--named", "b2="]);
    insta::assert_snapshot!(output, @r"
    ------- stderr -------
    Error: Argument 'b2=' must have the form NAME=REVISION, with both NAME and REVISION non-empty
    Hint: For example, `--named myfeature=@` is valid syntax
    [EOF]
    [exit status: 2]
    ");
    // test pushing a change with no equals sign
    let output = work_dir.run_jj(["git", "push", "--named", "b2"]);
    insta::assert_snapshot!(output, @r"
    ------- stderr -------
    Error: Argument 'b2' must include '=' and have the form NAME=REVISION
    Hint: For example, `--named myfeature=@` is valid syntax
    [EOF]
    [exit status: 2]
    ");

    // test pushing the same change with the same name again
    let output = work_dir.run_jj(["git", "push", "--named", "b1=@"]);
    insta::assert_snapshot!(output, @r"
    ------- stderr -------
    Error: Bookmark already exists: b1
    Hint: Use 'jj bookmark move' to move it, and 'jj git push -b b1 [--allow-new]' to push it
    [EOF]
    [exit status: 1]
    ");
    // test pushing two changes at once
    work_dir.write_file("file", "modified2");
    let output = work_dir.run_jj(["git", "push", "--named=b2=all:(@|@-)"]);
    insta::assert_snapshot!(output, @r"
    ------- stderr -------
    Error: Revset `all:(@|@-)` resolved to more than one revision
    Hint: The revset `all:(@|@-)` resolved to these revisions:
      yostqsxw 1b2bd869 b1* | pushed
      yqosqzyt 0f8164cd foo
    [EOF]
    [exit status: 1]
    ");

    // specifying the same bookmark with --named/--bookmark
    work_dir.write_file("file", "modified4");
    let output = work_dir.run_jj(["git", "push", "--named=b2=@", "-b=b2"]);
    insta::assert_snapshot!(output, @r"
    ------- stderr -------
    Changes to push to origin:
      Add bookmark b2 to 95ba7bdacb38
    [EOF]
    ");
}

#[test]
fn test_git_push_changes_with_name_deleted_tracked() {
    let test_env = TestEnvironment::default();
    set_up(&test_env);
    // Unset immutable_heads so that untracking branches does not move the working
    // copy
    test_env.add_config(r#"revset-aliases."immutable_heads()" = "none()""#);
    let work_dir = test_env.work_dir("local");
    // Create a second empty remote `another_remote`
    test_env
        .run_jj_in(".", ["git", "init", "another_remote"])
        .success();
    let another_remote_git_repo_path =
        git_repo_dir_for_jj_repo(&test_env.work_dir("another_remote"));
    work_dir
        .run_jj([
            "git",
            "remote",
            "add",
            "another_remote",
            another_remote_git_repo_path.to_str().unwrap(),
        ])
        .success();
    work_dir.run_jj(["describe", "-m", "foo"]).success();
    work_dir.write_file("file", "contents");
    work_dir.run_jj(["new", "-m", "pushed"]).success();
    work_dir.write_file("file", "modified");
    // Normal push as part of the test setup
    let output = work_dir.run_jj(["git", "push", "--named", "b1=@"]);
    insta::assert_snapshot!(output, @r"
    ------- stderr -------
    Changes to push to origin:
      Add bookmark b1 to 08f401c17d51
    [EOF]
    ");
    work_dir.run_jj(["bookmark", "delete", "b1"]).success();

    // Test the setup
    let output = work_dir
        .run_jj(["bookmark", "list", "--all", "b1"])
        .success();
    insta::assert_snapshot!(output, @r"
    b1 (deleted)
      @origin: kpqxywon 08f401c1 pushed
    [EOF]
    ------- stderr -------
    Hint: Bookmarks marked as deleted can be *deleted permanently* on the remote by running `jj git push --deleted`. Use `jj bookmark forget` if you don't want that.
    [EOF]
    ");

    // Can't push `b1` with --named to the same or another remote if it's deleted
    // locally and still tracked on `origin`
    let output = work_dir.run_jj(["git", "push", "--named", "b1=@", "--remote=another_remote"]);
    insta::assert_snapshot!(output, @r"
    ------- stderr -------
    Error: Tracked remote bookmarks exist for deleted bookmark: b1
    Hint: Use `jj bookmark set` to recreate the local bookmark. Run `jj bookmark untrack 'glob:b1@*'` to disassociate them.
    [EOF]
    [exit status: 1]
    ");
    let output = work_dir.run_jj(["git", "push", "--named", "b1=@", "--remote=origin"]);
    insta::assert_snapshot!(output, @r"
    ------- stderr -------
    Error: Tracked remote bookmarks exist for deleted bookmark: b1
    Hint: Use `jj bookmark set` to recreate the local bookmark. Run `jj bookmark untrack 'glob:b1@*'` to disassociate them.
    [EOF]
    [exit status: 1]
    ");

    // OK to push to a different remote once the bookmark is no longer tracked on
    // `origin`
    work_dir
        .run_jj(["bookmark", "untrack", "b1@origin"])
        .success();
    let output = work_dir
        .run_jj(["bookmark", "list", "--all", "b1"])
        .success();
    insta::assert_snapshot!(output, @r"
    b1@origin: kpqxywon 08f401c1 pushed
    [EOF]
    ");
    let output = work_dir.run_jj(["git", "push", "--named", "b1=@", "--remote=another_remote"]);
    insta::assert_snapshot!(output, @r"
    ------- stderr -------
    Changes to push to another_remote:
      Add bookmark b1 to 08f401c17d51
    [EOF]
    ");
    let output = work_dir
        .run_jj(["bookmark", "list", "--all", "b1"])
        .success();
    insta::assert_snapshot!(output, @r"
    b1: kpqxywon 08f401c1 pushed
      @another_remote: kpqxywon 08f401c1 pushed
    b1@origin: kpqxywon 08f401c1 pushed
    [EOF]
    ");
}

#[test]
fn test_git_push_changes_with_name_untracked_or_forgotten() {
    let test_env = TestEnvironment::default();
    set_up(&test_env);
    let work_dir = test_env.work_dir("local");
    // Unset immutable_heads so that untracking branches does not move the working
    // copy
    test_env.add_config(r#"revset-aliases."immutable_heads()" = "none()""#);
    work_dir.run_jj(["describe", "-m", "parent"]).success();
    work_dir.run_jj(["new", "-m", "pushed_to_remote"]).success();
    work_dir.write_file("file", "contents");
    work_dir
        .run_jj(["new", "-m", "child", "--no-edit"])
        .success();
    work_dir.write_file("file", "modified");

    // Push a branch to a remote, but forget the local branch
    work_dir
        .run_jj(["git", "push", "--named", "b1=@"])
        .success();
    work_dir
        .run_jj(["bookmark", "untrack", "b1@origin"])
        .success();
    work_dir.run_jj(["bookmark", "delete", "b1"]).success();

    let output = work_dir
        .run_jj(&[
            "log",
            "-r=::@+",
            r#"-T=separate(" ", commit_id.shortest(3), bookmarks, description)"#,
        ])
        .success();
    insta::assert_snapshot!(output, @r"
    ○  9a0 child
    @  767 b1@origin pushed_to_remote
    ○  aa9 parent
    ◆  000
    [EOF]
    ");
    let output = work_dir
        .run_jj(["bookmark", "list", "--all", "b1"])
        .success();
    insta::assert_snapshot!(output, @r"
    b1@origin: yostqsxw 767b63a5 pushed_to_remote
    [EOF]
    ");

    let output = work_dir.run_jj(["git", "push", "--named", "b1=@"]);
    insta::assert_snapshot!(output, @r"
    ------- stderr -------
    Error: Non-tracking remote bookmark b1@origin exists
    Hint: Run `jj bookmark track b1@origin` to import the remote bookmark.
    [EOF]
    [exit status: 1]
    ");

    let output = work_dir.run_jj(["git", "push", "--named", "b1=@+"]);
    insta::assert_snapshot!(output, @r"
    ------- stderr -------
    Error: Non-tracking remote bookmark b1@origin exists
    Hint: Run `jj bookmark track b1@origin` to import the remote bookmark.
    [EOF]
    [exit status: 1]
    ");

    // The bookmarked is still pushed to the remote, but let's entirely forget
    // it. In other words, let's forget the remote-tracking bookmarks.
    work_dir
        .run_jj(&["bookmark", "forget", "b1", "--include-remotes"])
        .success();
    let output = work_dir
        .run_jj(["bookmark", "list", "--all", "b1"])
        .success();
    insta::assert_snapshot!(output, @"");

    // Make sure push still errors if we try to push a bookmark with the same name
    // to a different location.
    let output = work_dir.run_jj(["git", "push", "--named", "b1=@-"]);
    insta::assert_snapshot!(output, @r"
    ------- stderr -------
    Changes to push to origin:
      Add bookmark b1 to aa9ad64cb4ce
    Error: Failed to push some bookmarks
    Hint: The following references unexpectedly moved on the remote:
      refs/heads/b1 (reason: stale info)
    Hint: Try fetching from the remote, then make the bookmark point to where you want it to be, and push again.
    [EOF]
    [exit status: 1]
    ");

    // The bookmark is still forgotten
    let output = work_dir.run_jj(["bookmark", "list", "--all", "b1"]);
    insta::assert_snapshot!(output, @"");
    let output = work_dir.run_jj(["git", "push", "--named", "b1=@+"]);
    insta::assert_snapshot!(output, @r"
    ------- stderr -------
    Changes to push to origin:
      Add bookmark b1 to 9a0f76645905
    Error: Failed to push some bookmarks
    Hint: The following references unexpectedly moved on the remote:
      refs/heads/b1 (reason: stale info)
    Hint: Try fetching from the remote, then make the bookmark point to where you want it to be, and push again.
    [EOF]
    [exit status: 1]
    ");
    // In this case, pushing the bookmark to the same location where it already is
    // succeeds. TODO: This seems pretty safe, but perhaps it should still show
    // an error or some sort of warning?
    let output = work_dir.run_jj(["git", "push", "--named", "b1=@"]);
    insta::assert_snapshot!(output, @r"
    ------- stderr -------
    Changes to push to origin:
      Add bookmark b1 to 767b63a598e1
    [EOF]
    ");
}

#[test]
fn test_git_push_revisions() {
    let test_env = TestEnvironment::default();
    set_up(&test_env);
    let work_dir = test_env.work_dir("local");
    work_dir.run_jj(["describe", "-m", "foo"]).success();
    work_dir.write_file("file", "contents");
    work_dir.run_jj(["new", "-m", "bar"]).success();
    work_dir
        .run_jj(["bookmark", "create", "-r@", "bookmark-1"])
        .success();
    work_dir.write_file("file", "modified");
    work_dir.run_jj(["new", "-m", "baz"]).success();
    work_dir
        .run_jj(["bookmark", "create", "-r@", "bookmark-2a"])
        .success();
    work_dir
        .run_jj(["bookmark", "create", "-r@", "bookmark-2b"])
        .success();
    work_dir.write_file("file", "modified again");

    // Push an empty set
    let output = work_dir.run_jj(["git", "push", "--allow-new", "-r=none()"]);
    insta::assert_snapshot!(output, @r"
    ------- stderr -------
    Warning: No bookmarks point to the specified revisions: none()
    Nothing changed.
    [EOF]
    ");
    // Push a revision with no bookmarks
    let output = work_dir.run_jj(["git", "push", "--allow-new", "-r=@--"]);
    insta::assert_snapshot!(output, @r"
    ------- stderr -------
    Warning: No bookmarks point to the specified revisions: @--
    Nothing changed.
    [EOF]
    ");
    // Push a revision with a single bookmark
    let output = work_dir.run_jj(["git", "push", "--allow-new", "-r=@-", "--dry-run"]);
    insta::assert_snapshot!(output, @r"
    ------- stderr -------
    Changes to push to origin:
      Add bookmark bookmark-1 to e76139e55e1e
    Dry-run requested, not pushing.
    [EOF]
    ");
    // Push multiple revisions of which some have bookmarks
    let output = work_dir.run_jj(["git", "push", "--allow-new", "-r=@--", "-r=@-", "--dry-run"]);
    insta::assert_snapshot!(output, @r"
    ------- stderr -------
    Warning: No bookmarks point to the specified revisions: @--
    Changes to push to origin:
      Add bookmark bookmark-1 to e76139e55e1e
    Dry-run requested, not pushing.
    [EOF]
    ");
    // Push a revision with a multiple bookmarks
    let output = work_dir.run_jj(["git", "push", "--allow-new", "-r=@", "--dry-run"]);
    insta::assert_snapshot!(output, @r"
    ------- stderr -------
    Changes to push to origin:
      Add bookmark bookmark-2a to 57d822f901bb
      Add bookmark bookmark-2b to 57d822f901bb
    Dry-run requested, not pushing.
    [EOF]
    ");
    // Repeating a commit doesn't result in repeated messages about the bookmark
    let output = work_dir.run_jj(["git", "push", "--allow-new", "-r=@-", "-r=@-", "--dry-run"]);
    insta::assert_snapshot!(output, @r"
    ------- stderr -------
    Changes to push to origin:
      Add bookmark bookmark-1 to e76139e55e1e
    Dry-run requested, not pushing.
    [EOF]
    ");
}

#[test]
fn test_git_push_mixed() {
    let test_env = TestEnvironment::default();
    set_up(&test_env);
    let work_dir = test_env.work_dir("local");
    work_dir.run_jj(["describe", "-m", "foo"]).success();
    work_dir.write_file("file", "contents");
    work_dir.run_jj(["new", "-m", "bar"]).success();
    work_dir
        .run_jj(["bookmark", "create", "-r@", "bookmark-1"])
        .success();
    work_dir.write_file("file", "modified");
    work_dir.run_jj(["new", "-m", "baz"]).success();
    work_dir
        .run_jj(["bookmark", "create", "-r@", "bookmark-2a"])
        .success();
    work_dir
        .run_jj(["bookmark", "create", "-r@", "bookmark-2b"])
        .success();
    work_dir.write_file("file", "modified again");

    // --allow-new is not implied for --bookmark=.. and -r=..
    let output = work_dir.run_jj([
        "git",
        "push",
        "--change=@--",
        "--bookmark=bookmark-1",
        "-r=@",
    ]);
    insta::assert_snapshot!(output, @r"
    ------- stderr -------
    Creating bookmark push-yqosqzytrlsw for revision yqosqzytrlsw
    Error: Refusing to create new remote bookmark bookmark-1@origin
    Hint: Use --allow-new to push new bookmark. Use --remote to specify the remote to push to.
    [EOF]
    [exit status: 1]
    ");

    let output = work_dir.run_jj([
        "git",
        "push",
        "--allow-new",
        "--change=@--",
        "--bookmark=bookmark-1",
        "-r=@",
    ]);
    insta::assert_snapshot!(output, @r"
    ------- stderr -------
    Creating bookmark push-yqosqzytrlsw for revision yqosqzytrlsw
    Changes to push to origin:
      Add bookmark push-yqosqzytrlsw to 0f8164cd580b
      Add bookmark bookmark-1 to e76139e55e1e
      Add bookmark bookmark-2a to 57d822f901bb
      Add bookmark bookmark-2b to 57d822f901bb
    [EOF]
    ");
}

#[test]
fn test_git_push_unsnapshotted_change() {
    let test_env = TestEnvironment::default();
    set_up(&test_env);
    let work_dir = test_env.work_dir("local");
    work_dir.run_jj(["describe", "-m", "foo"]).success();
    work_dir.write_file("file", "contents");
    work_dir.run_jj(["git", "push", "--change", "@"]).success();
    work_dir.write_file("file", "modified");
    work_dir.run_jj(["git", "push", "--change", "@"]).success();
}

#[test]
fn test_git_push_conflict() {
    let test_env = TestEnvironment::default();
    set_up(&test_env);
    let work_dir = test_env.work_dir("local");
    work_dir.write_file("file", "first");
    work_dir.run_jj(["commit", "-m", "first"]).success();
    work_dir.write_file("file", "second");
    work_dir.run_jj(["commit", "-m", "second"]).success();
    work_dir.write_file("file", "third");
    work_dir
        .run_jj(["rebase", "-r", "@", "-d", "@--"])
        .success();
    work_dir
        .run_jj(["bookmark", "create", "-r@", "my-bookmark"])
        .success();
    work_dir.run_jj(["describe", "-m", "third"]).success();
    let output = work_dir.run_jj(["git", "push", "--all"]);
    insta::assert_snapshot!(output, @r"
    ------- stderr -------
    Error: Won't push commit 654e715becca since it has conflicts
    Hint: Rejected commit: yostqsxw 654e715b my-bookmark | (conflict) third
    [EOF]
    [exit status: 1]
    ");
}

#[test]
fn test_git_push_no_description() {
    let test_env = TestEnvironment::default();
    set_up(&test_env);
    let work_dir = test_env.work_dir("local");
    work_dir
        .run_jj(["bookmark", "create", "-r@", "my-bookmark"])
        .success();
    work_dir.run_jj(["describe", "-m="]).success();
    let output = work_dir.run_jj(["git", "push", "--allow-new", "--bookmark", "my-bookmark"]);
    insta::assert_snapshot!(output, @r"
    ------- stderr -------
    Error: Won't push commit 8d23abddc924 since it has no description
    Hint: Rejected commit: yqosqzyt 8d23abdd my-bookmark | (empty) (no description set)
    [EOF]
    [exit status: 1]
    ");
    work_dir
        .run_jj([
            "git",
            "push",
            "--allow-new",
            "--bookmark",
            "my-bookmark",
            "--allow-empty-description",
        ])
        .success();
}

#[test]
fn test_git_push_no_description_in_immutable() {
    let test_env = TestEnvironment::default();
    set_up(&test_env);
    let work_dir = test_env.work_dir("local");
    work_dir
        .run_jj(["bookmark", "create", "-r@", "imm"])
        .success();
    work_dir.run_jj(["describe", "-m="]).success();
    work_dir.run_jj(["new", "-m", "foo"]).success();
    work_dir.write_file("file", "contents");
    work_dir
        .run_jj(["bookmark", "create", "-r@", "my-bookmark"])
        .success();

    let output = work_dir.run_jj([
        "git",
        "push",
        "--allow-new",
        "--bookmark=my-bookmark",
        "--dry-run",
    ]);
    insta::assert_snapshot!(output, @r"
    ------- stderr -------
    Error: Won't push commit 8d23abddc924 since it has no description
    Hint: Rejected commit: yqosqzyt 8d23abdd imm | (empty) (no description set)
    [EOF]
    [exit status: 1]
    ");

    test_env.add_config(r#"revset-aliases."immutable_heads()" = "imm""#);
    let output = work_dir.run_jj([
        "git",
        "push",
        "--allow-new",
        "--bookmark=my-bookmark",
        "--dry-run",
    ]);
    insta::assert_snapshot!(output, @r"
    ------- stderr -------
    Changes to push to origin:
      Add bookmark my-bookmark to 240e2e89abb2
    Dry-run requested, not pushing.
    [EOF]
    ");
}

#[test]
fn test_git_push_missing_author() {
    let test_env = TestEnvironment::default();
    set_up(&test_env);
    let work_dir = test_env.work_dir("local");
    let run_without_var = |var: &str, args: &[&str]| {
        work_dir
            .run_jj_with(|cmd| cmd.args(args).env_remove(var))
            .success();
    };
    run_without_var("JJ_USER", &["new", "root()", "-m=initial"]);
    run_without_var("JJ_USER", &["bookmark", "create", "-r@", "missing-name"]);
    let output = work_dir.run_jj(["git", "push", "--allow-new", "--bookmark", "missing-name"]);
    insta::assert_snapshot!(output, @r"
    ------- stderr -------
    Error: Won't push commit 613adaba9d49 since it has no author and/or committer set
    Hint: Rejected commit: vruxwmqv 613adaba missing-name | (empty) initial
    [EOF]
    [exit status: 1]
    ");
    run_without_var("JJ_EMAIL", &["new", "root()", "-m=initial"]);
    run_without_var("JJ_EMAIL", &["bookmark", "create", "-r@", "missing-email"]);
    let output = work_dir.run_jj(["git", "push", "--allow-new", "--bookmark=missing-email"]);
    insta::assert_snapshot!(output, @r"
    ------- stderr -------
    Error: Won't push commit bb4ea60fc9ba since it has no author and/or committer set
    Hint: Rejected commit: kpqxywon bb4ea60f missing-email | (empty) initial
    [EOF]
    [exit status: 1]
    ");
}

#[test]
fn test_git_push_missing_author_in_immutable() {
    let test_env = TestEnvironment::default();
    set_up(&test_env);
    let work_dir = test_env.work_dir("local");
    let run_without_var = |var: &str, args: &[&str]| {
        work_dir
            .run_jj_with(|cmd| cmd.args(args).env_remove(var))
            .success();
    };
    run_without_var("JJ_USER", &["new", "root()", "-m=no author name"]);
    run_without_var("JJ_EMAIL", &["new", "-m=no author email"]);
    work_dir
        .run_jj(["bookmark", "create", "-r@", "imm"])
        .success();
    work_dir.run_jj(["new", "-m", "foo"]).success();
    work_dir.write_file("file", "contents");
    work_dir
        .run_jj(["bookmark", "create", "-r@", "my-bookmark"])
        .success();

    let output = work_dir.run_jj([
        "git",
        "push",
        "--allow-new",
        "--bookmark=my-bookmark",
        "--dry-run",
    ]);
    insta::assert_snapshot!(output, @r"
    ------- stderr -------
    Error: Won't push commit 5c3cc711907f since it has no author and/or committer set
    Hint: Rejected commit: yostqsxw 5c3cc711 imm | (empty) no author email
    [EOF]
    [exit status: 1]
    ");

    test_env.add_config(r#"revset-aliases."immutable_heads()" = "imm""#);
    let output = work_dir.run_jj([
        "git",
        "push",
        "--allow-new",
        "--bookmark=my-bookmark",
        "--dry-run",
    ]);
    insta::assert_snapshot!(output, @r"
    ------- stderr -------
    Changes to push to origin:
      Add bookmark my-bookmark to 96080b93b4ce
    Dry-run requested, not pushing.
    [EOF]
    ");
}

#[test]
fn test_git_push_missing_committer() {
    let test_env = TestEnvironment::default();
    set_up(&test_env);
    let work_dir = test_env.work_dir("local");
    let run_without_var = |var: &str, args: &[&str]| {
        work_dir
            .run_jj_with(|cmd| cmd.args(args).env_remove(var))
            .success();
    };
    work_dir
        .run_jj(["bookmark", "create", "-r@", "missing-name"])
        .success();
    run_without_var("JJ_USER", &["describe", "-m=no committer name"]);
    let output = work_dir.run_jj(["git", "push", "--allow-new", "--bookmark=missing-name"]);
    insta::assert_snapshot!(output, @r"
    ------- stderr -------
    Error: Won't push commit e8a77cb24da9 since it has no author and/or committer set
    Hint: Rejected commit: yqosqzyt e8a77cb2 missing-name | (empty) no committer name
    [EOF]
    [exit status: 1]
    ");
    work_dir.run_jj(["new", "root()"]).success();
    work_dir
        .run_jj(["bookmark", "create", "-r@", "missing-email"])
        .success();
    run_without_var("JJ_EMAIL", &["describe", "-m=no committer email"]);
    let output = work_dir.run_jj(["git", "push", "--allow-new", "--bookmark=missing-email"]);
    insta::assert_snapshot!(output, @r"
    ------- stderr -------
    Error: Won't push commit 971c50fd8d1d since it has no author and/or committer set
    Hint: Rejected commit: kpqxywon 971c50fd missing-email | (empty) no committer email
    [EOF]
    [exit status: 1]
    ");

    // Test message when there are multiple reasons (missing committer and
    // description)
    run_without_var("JJ_EMAIL", &["describe", "-m=", "missing-email"]);
    let output = work_dir.run_jj(["git", "push", "--allow-new", "--bookmark=missing-email"]);
    insta::assert_snapshot!(output, @r"
    ------- stderr -------
    Error: Won't push commit 4bd3b55c7759 since it has no description and it has no author and/or committer set
    Hint: Rejected commit: kpqxywon 4bd3b55c missing-email | (empty) (no description set)
    [EOF]
    [exit status: 1]
    ");
}

#[test]
fn test_git_push_missing_committer_in_immutable() {
    let test_env = TestEnvironment::default();
    set_up(&test_env);
    let work_dir = test_env.work_dir("local");
    let run_without_var = |var: &str, args: &[&str]| {
        work_dir
            .run_jj_with(|cmd| cmd.args(args).env_remove(var))
            .success();
    };
    run_without_var("JJ_USER", &["describe", "-m=no committer name"]);
    work_dir.run_jj(["new"]).success();
    run_without_var("JJ_EMAIL", &["describe", "-m=no committer email"]);
    work_dir
        .run_jj(["bookmark", "create", "-r@", "imm"])
        .success();
    work_dir.run_jj(["new", "-m", "foo"]).success();
    work_dir.write_file("file", "contents");
    work_dir
        .run_jj(["bookmark", "create", "-r@", "my-bookmark"])
        .success();

    let output = work_dir.run_jj([
        "git",
        "push",
        "--allow-new",
        "--bookmark=my-bookmark",
        "--dry-run",
    ]);
    insta::assert_snapshot!(output, @r"
    ------- stderr -------
    Error: Won't push commit ab230f98c812 since it has no author and/or committer set
    Hint: Rejected commit: yostqsxw ab230f98 imm | (empty) no committer email
    [EOF]
    [exit status: 1]
    ");

    test_env.add_config(r#"revset-aliases."immutable_heads()" = "imm""#);
    let output = work_dir.run_jj([
        "git",
        "push",
        "--allow-new",
        "--bookmark=my-bookmark",
        "--dry-run",
    ]);
    insta::assert_snapshot!(output, @r"
    ------- stderr -------
    Changes to push to origin:
      Add bookmark my-bookmark to e0dff9c29479
    Dry-run requested, not pushing.
    [EOF]
    ");
}

#[test]
fn test_git_push_deleted() {
    let test_env = TestEnvironment::default();
    set_up(&test_env);
    let work_dir = test_env.work_dir("local");

    work_dir
        .run_jj(["bookmark", "delete", "bookmark1"])
        .success();
    let output = work_dir.run_jj(["git", "push", "--deleted"]);
    insta::assert_snapshot!(output, @r"
    ------- stderr -------
    Changes to push to origin:
      Delete bookmark bookmark1 from 9b2e76de3920
    [EOF]
    ");
    let output = work_dir.run_jj(["log", "-rall()"]);
    insta::assert_snapshot!(output, @r"
    @  yqosqzyt test.user@example.com 2001-02-03 08:05:13 8d23abdd
    │  (empty) (no description set)
    │ ○  zsuskuln test.user@example.com 2001-02-03 08:05:10 bookmark2 38a20473
    ├─╯  (empty) description 2
    │ ○  qpvuntsm test.user@example.com 2001-02-03 08:05:08 9b2e76de
    ├─╯  (empty) description 1
    ◆  zzzzzzzz root() 00000000
    [EOF]
    ");
    let output = work_dir.run_jj(["git", "push", "--deleted"]);
    insta::assert_snapshot!(output, @r"
    ------- stderr -------
    Nothing changed.
    [EOF]
    ");
}

#[test]
fn test_git_push_conflicting_bookmarks() {
    let test_env = TestEnvironment::default();
    set_up(&test_env);
    let work_dir = test_env.work_dir("local");
    test_env.add_config("git.auto-local-bookmark = true");
    let git_repo = {
        let mut git_repo_path = work_dir.root().to_owned();
        git_repo_path.extend([".jj", "repo", "store", "git"]);
        git::open(&git_repo_path)
    };

    // Forget remote ref, move local ref, then fetch to create conflict.
    git_repo
        .find_reference("refs/remotes/origin/bookmark2")
        .unwrap()
        .delete()
        .unwrap();
    work_dir.run_jj(["git", "import"]).success();
    work_dir
        .run_jj(["new", "root()", "-m=description 3"])
        .success();
    work_dir
        .run_jj(["bookmark", "create", "-r@", "bookmark2"])
        .success();
    work_dir.run_jj(["git", "fetch"]).success();
    insta::assert_snapshot!(get_bookmark_output(&work_dir), @r"
    bookmark1: qpvuntsm 9b2e76de (empty) description 1
      @origin: qpvuntsm 9b2e76de (empty) description 1
    bookmark2 (conflicted):
      + yostqsxw ebedbe63 (empty) description 3
      + zsuskuln 38a20473 (empty) description 2
      @origin (behind by 1 commits): zsuskuln 38a20473 (empty) description 2
    [EOF]
    ");

    let bump_bookmark1 = || {
        work_dir.run_jj(["new", "bookmark1", "-m=bump"]).success();
        work_dir
            .run_jj(["bookmark", "set", "bookmark1", "-r@"])
            .success();
    };

    // Conflicting bookmark at @
    let output = work_dir.run_jj(["git", "push", "--allow-new"]);
    insta::assert_snapshot!(output, @r"
    ------- stderr -------
    Warning: Bookmark bookmark2 is conflicted
    Hint: Run `jj bookmark list` to inspect, and use `jj bookmark set` to fix it up.
    Nothing changed.
    [EOF]
    ");

    // --bookmark should be blocked by conflicting bookmark
    let output = work_dir.run_jj(["git", "push", "--allow-new", "--bookmark", "bookmark2"]);
    insta::assert_snapshot!(output, @r"
    ------- stderr -------
    Error: Bookmark bookmark2 is conflicted
    Hint: Run `jj bookmark list` to inspect, and use `jj bookmark set` to fix it up.
    [EOF]
    [exit status: 1]
    ");

    // --all shouldn't be blocked by conflicting bookmark
    bump_bookmark1();
    let output = work_dir.run_jj(["git", "push", "--all"]);
    insta::assert_snapshot!(output, @r"
    ------- stderr -------
    Warning: Bookmark bookmark2 is conflicted
    Hint: Run `jj bookmark list` to inspect, and use `jj bookmark set` to fix it up.
    Changes to push to origin:
      Move forward bookmark bookmark1 from 9b2e76de3920 to 749c2e6d999f
    [EOF]
    ");

    // --revisions shouldn't be blocked by conflicting bookmark
    bump_bookmark1();
    let output = work_dir.run_jj(["git", "push", "--allow-new", "-rall()"]);
    insta::assert_snapshot!(output, @r"
    ------- stderr -------
    Warning: Bookmark bookmark2 is conflicted
    Hint: Run `jj bookmark list` to inspect, and use `jj bookmark set` to fix it up.
    Changes to push to origin:
      Move forward bookmark bookmark1 from 749c2e6d999f to 9bb0f427b517
    [EOF]
    ");
}

#[test]
fn test_git_push_deleted_untracked() {
    let test_env = TestEnvironment::default();
    set_up(&test_env);
    let work_dir = test_env.work_dir("local");

    // Absent local bookmark shouldn't be considered "deleted" compared to
    // non-tracking remote bookmark.
    work_dir
        .run_jj(["bookmark", "delete", "bookmark1"])
        .success();
    work_dir
        .run_jj(["bookmark", "untrack", "bookmark1@origin"])
        .success();
    let output = work_dir.run_jj(["git", "push", "--deleted"]);
    insta::assert_snapshot!(output, @r"
    ------- stderr -------
    Nothing changed.
    [EOF]
    ");
    let output = work_dir.run_jj(["git", "push", "--bookmark=bookmark1"]);
    insta::assert_snapshot!(output, @r"
    ------- stderr -------
    Error: No such bookmark: bookmark1
    [EOF]
    [exit status: 1]
    ");
}

#[test]
fn test_git_push_tracked_vs_all() {
    let test_env = TestEnvironment::default();
    set_up(&test_env);
    let work_dir = test_env.work_dir("local");
    work_dir
        .run_jj(["new", "bookmark1", "-mmoved bookmark1"])
        .success();
    work_dir
        .run_jj(["bookmark", "set", "bookmark1", "-r@"])
        .success();
    work_dir
        .run_jj(["new", "bookmark2", "-mmoved bookmark2"])
        .success();
    work_dir
        .run_jj(["bookmark", "delete", "bookmark2"])
        .success();
    work_dir
        .run_jj(["bookmark", "untrack", "bookmark1@origin"])
        .success();
    work_dir
        .run_jj(["bookmark", "create", "-r@", "bookmark3"])
        .success();
    insta::assert_snapshot!(get_bookmark_output(&work_dir), @r"
    bookmark1: vruxwmqv d7607a25 (empty) moved bookmark1
    bookmark1@origin: qpvuntsm 9b2e76de (empty) description 1
    bookmark2 (deleted)
      @origin: zsuskuln 38a20473 (empty) description 2
    bookmark3: znkkpsqq 0004a65e (empty) moved bookmark2
    [EOF]
    ");

    // At this point, only bookmark2 is still tracked.
    // `jj git push --tracked --deleted` would try to push it and no other
    // bookmarks.
    let output = work_dir.run_jj(["git", "push", "--tracked", "--dry-run"]);
    insta::assert_snapshot!(output, @r"
    ------- stderr -------
    Warning: Refusing to push deleted bookmark bookmark2
    Hint: Push deleted bookmarks with --deleted or forget the bookmark to suppress this warning.
    Nothing changed.
    [EOF]
    ");
    let output = work_dir.run_jj(["git", "push", "--tracked", "--deleted", "--dry-run"]);
    insta::assert_snapshot!(output, @r"
    ------- stderr -------
    Changes to push to origin:
      Delete bookmark bookmark2 from 38a204733702
    Dry-run requested, not pushing.
    [EOF]
    ");

    // Untrack the last remaining tracked bookmark.
    work_dir
        .run_jj(["bookmark", "untrack", "bookmark2@origin"])
        .success();
    insta::assert_snapshot!(get_bookmark_output(&work_dir), @r"
    bookmark1: vruxwmqv d7607a25 (empty) moved bookmark1
    bookmark1@origin: qpvuntsm 9b2e76de (empty) description 1
    bookmark2@origin: zsuskuln 38a20473 (empty) description 2
    bookmark3: znkkpsqq 0004a65e (empty) moved bookmark2
    [EOF]
    ");

    // Now, no bookmarks are tracked. --tracked does not push anything
    let output = work_dir.run_jj(["git", "push", "--tracked"]);
    insta::assert_snapshot!(output, @r"
    ------- stderr -------
    Nothing changed.
    [EOF]
    ");

    // All bookmarks are still untracked.
    // - --all tries to push bookmark1, but fails because a bookmark with the same
    // name exist on the remote.
    // - --all succeeds in pushing bookmark3, since there is no bookmark of the same
    // name on the remote.
    // - It does not try to push bookmark2.
    //
    // TODO: Not trying to push bookmark2 could be considered correct, or perhaps
    // we want to consider this as a deletion of the bookmark that failed because
    // the bookmark was untracked. In the latter case, an error message should be
    // printed. Some considerations:
    // - Whatever we do should be consistent with what `jj bookmark list` does; it
    //   currently does *not* list bookmarks like bookmark2 as "about to be
    //   deleted", as can be seen above.
    // - We could consider showing some hint on `jj bookmark untrack
    //   bookmark2@origin` instead of showing an error here.
    let output = work_dir.run_jj(["git", "push", "--all"]);
    insta::assert_snapshot!(output, @r"
    ------- stderr -------
    Warning: Non-tracking remote bookmark bookmark1@origin exists
    Hint: Run `jj bookmark track bookmark1@origin` to import the remote bookmark.
    Changes to push to origin:
      Add bookmark bookmark3 to 0004a65e1d28
    [EOF]
    ");
}

#[test]
fn test_git_push_moved_forward_untracked() {
    let test_env = TestEnvironment::default();
    set_up(&test_env);
    let work_dir = test_env.work_dir("local");

    work_dir
        .run_jj(["new", "bookmark1", "-mmoved bookmark1"])
        .success();
    work_dir
        .run_jj(["bookmark", "set", "bookmark1", "-r@"])
        .success();
    work_dir
        .run_jj(["bookmark", "untrack", "bookmark1@origin"])
        .success();
    let output = work_dir.run_jj(["git", "push", "--allow-new"]);
    insta::assert_snapshot!(output, @r"
    ------- stderr -------
    Warning: Non-tracking remote bookmark bookmark1@origin exists
    Hint: Run `jj bookmark track bookmark1@origin` to import the remote bookmark.
    Nothing changed.
    [EOF]
    ");
}

#[test]
fn test_git_push_moved_sideways_untracked() {
    let test_env = TestEnvironment::default();
    set_up(&test_env);
    let work_dir = test_env.work_dir("local");

    work_dir
        .run_jj(["new", "root()", "-mmoved bookmark1"])
        .success();
    work_dir
        .run_jj(["bookmark", "set", "--allow-backwards", "bookmark1", "-r@"])
        .success();
    work_dir
        .run_jj(["bookmark", "untrack", "bookmark1@origin"])
        .success();
    let output = work_dir.run_jj(["git", "push", "--allow-new"]);
    insta::assert_snapshot!(output, @r"
    ------- stderr -------
    Warning: Non-tracking remote bookmark bookmark1@origin exists
    Hint: Run `jj bookmark track bookmark1@origin` to import the remote bookmark.
    Nothing changed.
    [EOF]
    ");
}

#[test]
fn test_git_push_to_remote_named_git() {
    let test_env = TestEnvironment::default();
    set_up(&test_env);
    let work_dir = test_env.work_dir("local");
    let git_repo_path = {
        let mut git_repo_path = work_dir.root().to_owned();
        git_repo_path.extend([".jj", "repo", "store", "git"]);
        git_repo_path
    };
    git::rename_remote(&git_repo_path, "origin", "git");

    let output = work_dir.run_jj(["git", "push", "--all", "--remote=git"]);
    insta::assert_snapshot!(output, @r"
    ------- stderr -------
    Changes to push to git:
      Add bookmark bookmark1 to 9b2e76de3920
      Add bookmark bookmark2 to 38a204733702
    Error: Git remote named 'git' is reserved for local Git repository
    Hint: Run `jj git remote rename` to give a different name.
    [EOF]
    [exit status: 1]
    ");
}

#[test]
fn test_git_push_to_remote_with_slashes() {
    let test_env = TestEnvironment::default();
    set_up(&test_env);
    let work_dir = test_env.work_dir("local");
    let git_repo_path = {
        let mut git_repo_path = work_dir.root().to_owned();
        git_repo_path.extend([".jj", "repo", "store", "git"]);
        git_repo_path
    };
    git::rename_remote(&git_repo_path, "origin", "slash/origin");

    let output = work_dir.run_jj(["git", "push", "--all", "--remote=slash/origin"]);
    insta::assert_snapshot!(output, @r"
    ------- stderr -------
    Changes to push to slash/origin:
      Add bookmark bookmark1 to 9b2e76de3920
      Add bookmark bookmark2 to 38a204733702
    Error: Git remotes with slashes are incompatible with jj: slash/origin
    Hint: Run `jj git remote rename` to give a different name.
    [EOF]
    [exit status: 1]
    ");
}

#[test]
fn test_git_push_sign_on_push() {
    let test_env = TestEnvironment::default();
    set_up(&test_env);
    let work_dir = test_env.work_dir("local");
    let template = r#"
    separate("\n",
      description.first_line(),
      if(signature,
        separate(", ",
          "Signature: " ++ signature.display(),
          "Status: " ++ signature.status(),
          "Key: " ++ signature.key(),
        )
      )
    )
    "#;
    work_dir
        .run_jj(["new", "bookmark2", "-m", "commit to be signed 1"])
        .success();
    work_dir
        .run_jj(["new", "-m", "commit to be signed 2"])
        .success();
    work_dir
        .run_jj(["bookmark", "set", "bookmark2", "-r@"])
        .success();
    work_dir
        .run_jj(["new", "-m", "commit which should not be signed 1"])
        .success();
    work_dir
        .run_jj(["new", "-m", "commit which should not be signed 2"])
        .success();
    // There should be no signed commits initially
    let output = work_dir.run_jj(["log", "-T", template]);
    insta::assert_snapshot!(output, @r"
    @  commit which should not be signed 2
    ○  commit which should not be signed 1
    ○  commit to be signed 2
    ○  commit to be signed 1
    ○  description 2
    │ ○  description 1
    ├─╯
    ◆
    [EOF]
    ");
    test_env.add_config(
        r#"
    signing.backend = "test"
    signing.key = "impeccable"
    git.sign-on-push = true
    "#,
    );
    let output = work_dir.run_jj(["git", "push", "--dry-run"]);
    insta::assert_snapshot!(output, @r"
    ------- stderr -------
    Changes to push to origin:
      Move forward bookmark bookmark2 from 38a204733702 to 3779ed7f18df
    Dry-run requested, not pushing.
    [EOF]
    ");
    // There should be no signed commits after performing a dry run
    let output = work_dir.run_jj(["log", "-T", template]);
    insta::assert_snapshot!(output, @r"
    @  commit which should not be signed 2
    ○  commit which should not be signed 1
    ○  commit to be signed 2
    ○  commit to be signed 1
    ○  description 2
    │ ○  description 1
    ├─╯
    ◆
    [EOF]
    ");
    let output = work_dir.run_jj(["git", "push"]);
    insta::assert_snapshot!(output, @r"
    ------- stderr -------
    Updated signatures of 2 commits
    Rebased 2 descendant commits
    Changes to push to origin:
      Move forward bookmark bookmark2 from 38a204733702 to d45e2adce0ad
    Working copy  (@) now at: kmkuslsw 3d5a9465 (empty) commit which should not be signed 2
    Parent commit (@-)      : kpqxywon 48ea83e9 (empty) commit which should not be signed 1
    [EOF]
    ");
    // Only commits which are being pushed should be signed
    let output = work_dir.run_jj(["log", "-T", template]);
    insta::assert_snapshot!(output, @r"
    @  commit which should not be signed 2
    ○  commit which should not be signed 1
    ○  commit to be signed 2
    │  Signature: test-display, Status: good, Key: impeccable
    ○  commit to be signed 1
    │  Signature: test-display, Status: good, Key: impeccable
    ○  description 2
    │ ○  description 1
    ├─╯
    ◆
    [EOF]
    ");

    // Immutable commits should not be signed
    let output = work_dir.run_jj([
        "bookmark",
        "create",
        "bookmark3",
        "-r",
        "description('commit which should not be signed 1')",
    ]);
    insta::assert_snapshot!(output, @r"
    ------- stderr -------
    Created 1 bookmarks pointing to kpqxywon 48ea83e9 bookmark3 | (empty) commit which should not be signed 1
    [EOF]
    ");
    let output = work_dir.run_jj(["bookmark", "move", "bookmark2", "--to", "bookmark3"]);
    insta::assert_snapshot!(output, @r"
    ------- stderr -------
    Moved 1 bookmarks to kpqxywon 48ea83e9 bookmark2* bookmark3 | (empty) commit which should not be signed 1
    [EOF]
    ");
    test_env.add_config(r#"revset-aliases."immutable_heads()" = "bookmark3""#);
    let output = work_dir.run_jj(["git", "push"]);
    insta::assert_snapshot!(output, @r"
    ------- stderr -------
    Warning: Refusing to create new remote bookmark bookmark3@origin
    Hint: Use --allow-new to push new bookmark. Use --remote to specify the remote to push to.
    Changes to push to origin:
      Move forward bookmark bookmark2 from d45e2adce0ad to 48ea83e9499c
    [EOF]
    ");
    let output = work_dir.run_jj(["log", "-T", template, "-r", "::"]);
    insta::assert_snapshot!(output, @r"
    @  commit which should not be signed 2
    ◆  commit which should not be signed 1
    ◆  commit to be signed 2
    │  Signature: test-display, Status: good, Key: impeccable
    ◆  commit to be signed 1
    │  Signature: test-display, Status: good, Key: impeccable
    ◆  description 2
    │ ○  description 1
    ├─╯
    ◆
    [EOF]
    ");
}

#[test]
fn test_git_push_rejected_by_remote() {
    let test_env = TestEnvironment::default();
    set_up(&test_env);
    let work_dir = test_env.work_dir("local");
    // show repo state
    insta::assert_snapshot!(get_bookmark_output(&work_dir), @r"
    bookmark1: qpvuntsm 9b2e76de (empty) description 1
      @origin: qpvuntsm 9b2e76de (empty) description 1
    bookmark2: zsuskuln 38a20473 (empty) description 2
      @origin: zsuskuln 38a20473 (empty) description 2
    [EOF]
    ");

    // create a hook on the remote that prevents pushing
    let hook_path = test_env
        .env_root()
        .join("origin")
        .join(".jj")
        .join("repo")
        .join("store")
        .join("git")
        .join("hooks")
        .join("update");

    std::fs::write(&hook_path, "#!/bin/sh\nexit 1").unwrap();
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt as _;

        std::fs::set_permissions(&hook_path, std::fs::Permissions::from_mode(0o700)).unwrap();
    }

    // create new commit on top of bookmark1
    work_dir.run_jj(["new", "bookmark1"]).success();
    work_dir.write_file("file", "file");
    work_dir.run_jj(["describe", "-m=update"]).success();

    // update bookmark
    work_dir.run_jj(["bookmark", "move", "bookmark1"]).success();

    // push bookmark
    let output = work_dir.run_jj(["git", "push"]);

    // The git remote sideband adds a dummy suffix of 8 spaces to attempt to clear
    // any leftover data. This is done to help with cases where the line is
    // rewritten.
    //
    // However, a common option in a lot of editors removes trailing whitespace.
    // This means that anyone with that option that opens this file would make the
    // following snapshot fail. Using the insta filter here normalizes the
    // output.
    let mut settings = insta::Settings::clone_current();
    settings.add_filter(r"\s*\n", "\n");
    settings.bind(|| {
        insta::assert_snapshot!(output, @r"
        ------- stderr -------
        Changes to push to origin:
          Move forward bookmark bookmark1 from 9b2e76de3920 to 0fc4cf312e83
        remote: error: hook declined to update refs/heads/bookmark1
        Error: Failed to push some bookmarks
        Hint: The remote rejected the following updates:
          refs/heads/bookmark1 (reason: hook declined)
        Hint: Try checking if you have permission to push to all the bookmarks.
        [EOF]
        [exit status: 1]
        ");
    });
}

// TODO: Remove with the `git.subprocess` setting.
#[test]
fn test_git_push_git2_warning() {
    let test_env = TestEnvironment::default();
    set_up(&test_env);
    let work_dir = test_env.work_dir("local");
    test_env.add_config("git.subprocess = false");
    work_dir
        .run_jj(["describe", "bookmark1", "-m", "modified bookmark1 commit"])
        .success();
    let output = work_dir.run_jj(["git", "push", "--all"]);
    if cfg!(feature = "git2") {
        insta::assert_snapshot!(output, @r"
        ------- stderr -------
        Warning: `git.subprocess = false` will be removed in 0.30; please report any issues you have with the default.
        Changes to push to origin:
          Move sideways bookmark bookmark1 from d13ecdbda2a2 to 0f8dc6560f32
        [EOF]
        ");
    } else {
        insta::assert_snapshot!(output, @r"
        ------- stderr -------
        Changes to push to origin:
          Move sideways bookmark bookmark1 from 9b2e76de3920 to e5ce6d9a0991
        [EOF]
        ");
    }
}

#[test]
fn test_git_push_custom_revset() {
    let test_env = TestEnvironment::default();
    set_up(&test_env);
    let work_dir = test_env.work_dir("local");
    // add a custom revset which simulates ignoring a `private()` revset.
    test_env.add_config(
        r#"
    [revsets]
    'push(remote)' = "remote_bookmarks(remote=remote)..@ & ~bookmarks(glob:'local/*') & ~subject(glob:'wip:*')"
    "#,
    );
    work_dir
        .run_jj(["new", "bookmark2", "-m", "commit to be pushed"])
        .success();
    work_dir.run_jj(["new", "-m", "wip: stuff"]).success();
    work_dir
        .run_jj(["bookmark", "set", "local/stuff", "-r@"])
        .success();
    work_dir
        .run_jj(["new", "-m", "commit which should pushed"])
        .success();
    work_dir
        .run_jj(["new", "-m", "wip: commit which should not be pushed"])
        .success();
    //
    let output = work_dir.run_jj(["log"]);
    insta::assert_snapshot!(output, @r"
    @  kmkuslsw test.user@example.com 2001-02-03 08:05:18 ead3bc40
    │  (empty) wip: commit which should not be pushed
    ○  kpqxywon test.user@example.com 2001-02-03 08:05:17 048a6554
    │  (empty) commit which should pushed
    ○  yostqsxw test.user@example.com 2001-02-03 08:05:15 local/stuff 0712d559
    │  (empty) wip: stuff
    ○  vruxwmqv test.user@example.com 2001-02-03 08:05:14 a615282a
    │  (empty) commit to be pushed
    ○  zsuskuln test.user@example.com 2001-02-03 08:05:10 bookmark2 38a20473
    │  (empty) description 2
    │ ○  qpvuntsm test.user@example.com 2001-02-03 08:05:08 bookmark1 9b2e76de
    ├─╯  (empty) description 1
    ◆  zzzzzzzz root() 00000000
    [EOF]
    ");
    // We should try to push the first two commits but not any containing "wip:".
    let output = work_dir.run_jj(["git", "push"]);
    insta::assert_snapshot!(output, @r"
    ------- stderr -------
    Nothing changed.
    [EOF]
    ");
}

#[must_use]
fn get_bookmark_output(work_dir: &TestWorkDir) -> CommandOutput {
    // --quiet to suppress deleted bookmarks hint
    work_dir.run_jj(["bookmark", "list", "--all-remotes", "--quiet"])
}
