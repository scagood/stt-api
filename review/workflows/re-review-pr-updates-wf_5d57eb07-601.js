export const meta = {
  name: 're-review-pr-updates',
  description: 'Check whether each PR fixed the posted review findings and review the new commits; adversarially verify anything still open',
  phases: [
    { title: 'Re-review', detail: 'per PR: status of each posted finding + new bugs in the new commits' },
    { title: 'Verify', detail: 'skeptic tries to refute every not-fixed / partial / new finding' },
  ],
}

const REPO = '/home/user/stt-api'
const SCRATCH = '/tmp/claude-0/-home-user-stt-api/349487e0-d09b-554f-9be8-0221f8cd653b/scratchpad'
const PY = SCRATCH + '/venv/bin/python'

const STATUS = {
  type: 'object',
  properties: {
    head: { type: 'string' },
    testsOnBranch: { type: 'string' },
    testsMergedWithMain: { type: 'string' },
    authorReplies: { type: 'string', description: 'summary of any replies / resolutions on the review threads or PR comments since the review; empty if none' },
    findings: {
      type: 'array',
      items: {
        type: 'object',
        properties: {
          id: { type: 'string', description: 'F1, F2, ... or N1.. for nits, in the order of the posted review' },
          title: { type: 'string' },
          status: { type: 'string', enum: ['fixed', 'partly_fixed', 'not_fixed', 'declined_with_reason', 'not_applicable'] },
          evidence: { type: 'string', description: 'what you ran (repro on new head) and what it showed; for declined, the author reason and whether it holds' },
          remaining: { type: 'string', description: 'what is still wrong, with a concrete repro; empty if fixed' },
        },
        required: ['id', 'title', 'status', 'evidence', 'remaining'],
      },
    },
    newIssues: {
      type: 'array',
      items: {
        type: 'object',
        properties: {
          severity: { type: 'string', enum: ['High', 'Medium', 'Low', 'Nit'] },
          title: { type: 'string' },
          file: { type: 'string' },
          line: { type: 'integer' },
          detail: { type: 'string', description: 'what is wrong, concrete repro input -> actual vs expected, suggested fix' },
        },
        required: ['severity', 'title', 'file', 'line', 'detail'],
      },
    },
    summary: { type: 'string' },
  },
  required: ['head', 'testsOnBranch', 'testsMergedWithMain', 'authorReplies', 'findings', 'newIssues', 'summary'],
}

const VERDICTS = {
  type: 'object',
  properties: {
    checks: {
      type: 'array',
      items: {
        type: 'object',
        properties: {
          ref: { type: 'string', description: 'finding id (F1..) or "NEW: <title>"' },
          verdict: { type: 'string', enum: ['confirmed', 'plausible', 'refuted'] },
          evidence: { type: 'string' },
          correction: { type: 'string', description: 'anything overstated or wrong in the claim; empty if none' },
        },
        required: ['ref', 'verdict', 'evidence', 'correction'],
      },
    },
    fixedChecks: {
      type: 'array',
      description: 'spot-check of findings the re-reviewer marked fixed: is the fix real?',
      items: {
        type: 'object',
        properties: { ref: { type: 'string' }, agree: { type: 'boolean' }, evidence: { type: 'string' } },
        required: ['ref', 'agree', 'evidence'],
      },
    },
  },
  required: ['checks', 'fixedChecks'],
}

const setup = (pr, tag) => `Setup:
- Repo: ${REPO}. PR branch fetched as origin/${pr.branch}. Reviewed head was ${pr.oldSha}; new head is ${pr.newSha}. Confirm with \`git -C ${REPO} ls-remote origin refs/heads/${pr.branch}\` (if it moved again, \`git -C ${REPO} fetch origin ${pr.branch}:refs/remotes/origin/${pr.branch}\` and use the newest, saying so).
- New commits: \`git -C ${REPO} log --oneline ${pr.oldSha}..origin/${pr.branch}\`, diff \`git -C ${REPO} diff ${pr.oldSha} origin/${pr.branch}\`.
- Own worktree: \`git -C ${REPO} worktree add --detach ${SCRATCH}/${tag}${pr.number} origin/${pr.branch}\`; remove at the end with \`git -C ${REPO} worktree remove --force ${SCRATCH}/${tag}${pr.number}\`. Throwaway scripts go in ${SCRATCH}/${tag}${pr.number}-scripts/. Never modify ${REPO} itself, never push, never post to GitHub.
- Python 3.14 venv with test deps: ${PY} (run from inside your worktree so imports hit the PR code). origin/main is current main.
- The posted review is ${pr.reviewUrl}. Read it and its inline threads (incl. any replies) with mcp__github__pull_request_read (load via ToolSearch "select:mcp__github__pull_request_read"; methods get_reviews, get_review_comments, get_comments, get). The posted text is the source of truth for what was asked.
${pr.context ? 'Context: ' + pr.context : ''}`

const reviewPrompt = pr => `Re-review GitHub PR scagood/stt-api#${pr.number} ("${pr.title}") after its author pushed changes in response to a code review. Read-only: no pushing, no GitHub writes.

${setup(pr, 'rr')}

Original findings (summary; the posted review has the exact wording, repros and corrections):
${pr.findings}

Do:
1. For EACH posted finding and nit, decide its status on the new head by re-running its repro (or the posted one) against the new code — don't trust the commit message. 'fixed' only if the repro now behaves correctly and the fix is complete for the cases named in the review (incl. any extra cases the posted corrections listed). 'declined_with_reason' if the author replied declining; judge whether the reason holds.
2. Review the NEW commits as fresh code: correctness bugs, edge cases, regressions vs the reviewed head and vs main, tests that don't test what they claim, docs/README/comments contradicting code. Verify each candidate concretely (script or exact trace); drop what you can't substantiate.
3. Run the full test suite on the branch, and on the branch merged with origin/main (in your worktree: \`git -c user.email=x@x -c user.name=x merge --no-edit origin/main\`). Report counts.
${pr.extra || ''}
Be concise in evidence strings but include the concrete inputs and outputs.`

const verifyPrompt = (pr, s) => `You are an independent skeptic. Another agent re-reviewed GitHub PR scagood/stt-api#${pr.number} ("${pr.title}") after its author pushed fixes, and produced the claims below. Try to REFUTE every claim that something is still wrong (not_fixed / partly_fixed / new issues), and spot-check the 'fixed' verdicts. Read-only: no pushing, no GitHub writes.

${setup(pr, 'rv')}

Re-reviewer's output:
${JSON.stringify(s, null, 2)}

For each finding with status not_fixed or partly_fixed, and each newIssue: reproduce it yourself on the new head. verdict 'refuted' only with concrete evidence it's wrong (e.g. the repro behaves correctly, the cited code doesn't do that, or the issue exists identically on main and the claim says regression); 'plausible' if you could not reproduce or disprove; 'confirmed' if your own run shows it. Put overstatements (severity, numbers, scope) in 'correction'. ref = finding id or "NEW: <title>".
For each finding marked fixed (or declined_with_reason), spot-check it: rerun the original repro on the new head and say if you agree.`

const results = await pipeline(
  args,
  pr => agent(reviewPrompt(pr), { label: `re-review:#${pr.number}`, phase: 'Re-review', schema: STATUS }),
  (s, pr) => {
    if (!s) { log(`#${pr.number}: re-review failed`); return null }
    const open = s.findings.filter(f => !['fixed', 'not_applicable'].includes(f.status)).length + s.newIssues.length
    log(`#${pr.number}: ${s.findings.filter(f => f.status === 'fixed').length}/${s.findings.length} fixed, ${s.newIssues.length} new issue(s); verifying ${open} open item(s)`)
    return agent(verifyPrompt(pr, s), { label: `verify:#${pr.number}`, phase: 'Verify', schema: VERDICTS })
      .then(v => ({ number: pr.number, status: s, verify: v }))
  },
)
return results.filter(Boolean)
