export const meta = {
  name: 'verify-78-fixes',
  description: 'Verify #78 fix commits (24f703c) against the review repros and main, for the merge decision',
  phases: [{ title: 'Verify', detail: 'repros, differential vs main, superset, tests' }],
}
const REPO = '/home/user/stt-api'
const SCRATCH = '/tmp/claude-0/-home-user-stt-api/349487e0-d09b-554f-9be8-0221f8cd653b/scratchpad'
const PY = SCRATCH + '/venv/bin/python'
const BR = 'fix/loud-frames-whole-runs'
phase('Verify')
return await agent(`Verify the fix commits on GitHub PR scagood/stt-api#78 before a merge decision. Read-only: no push, no GitHub writes, don't modify ${REPO}.
Setup: \`git -C ${REPO} fetch -q origin main ${BR}:refs/remotes/origin/${BR}\`; head should be 24f703c (commits f24610c and 24f703c on top of the reviewed 2aa0b34; base origin/main ed4be7f) — confirm with ls-remote, report full sha, use newest if moved and list any extra commits. Worktrees under ${SCRATCH}/v78-* (remove at end); scripts under ${SCRATCH}/v78-scripts/. Venv ${PY}. The earlier review's scripts are in ${SCRATCH}/k78a-scripts/ and ${SCRATCH}/k78b-scripts/ (reuse them; they contain the repros and the differential/fuzz harnesses). The posted review and threads: mcp__github__pull_request_read (ToolSearch "select:mcp__github__pull_request_read"; methods get_reviews, get_review_comments).
The change (parakeet_service/chunker.py loud_frames): join quiet sounds across dip = max(_SOUND_DIP=20 frames, int(VAD_MIN_SILENCE_MS/20)); kept = heard & near (frame-wise) drives re-listening (todo from runs of ~kept); whole sounds touching near are ORed into loud but only within reach = near extended by _HEARD_ENOUGH (0.5 s).
Check, comparing head vs main (ed4be7f) vs reviewed head 2aa0b34:
1. Medium fixed: the review's repro (volume VAD, gate unset, target 60, max 75, ctx 5; two 40 s turns at -20 dBFS around a 12 s pause over -55 dBFS; ten 0.35 s words at -42 dBFS 0.5 s apart from 41.0 s) at VAD_MIN_SILENCE_MS 500, 600, 1000: head must decode 10/10 like main. Differential on random quiet-speech pauses at 60, 100, 400, 600, 800, 1000, 3000 ms: count cases where head keeps/decodes fewer words or cuts more than main (must be 0, or explain each).
2. Item 1 still fixed: straddling-word repro (quiet phrase 45.0-46.0 at -40 dBFS, 0.4 s word 48.95-49.35, 3 s+ pause) -> head (44.88,49.48)-like, whole word, retime pause not inside it.
3. Retime Low fixed: the aside repro (retime ratio 0.6; host -20 dBFS, room -64; quiet guest 30.5-38.5 at -36; 0.3 s breath at -42 at 41.4-41.7; 0.8 s aside at -48 at 42.15-42.95; host at 44.55) -> pauses equal main's; retime leaves the aside in place.
4. Train-of-sounds Low: clicks repro (quiet 1 s phrase at -40 at 41-42; seven 60 ms clicks at -44 every 0.36 s 44.8-47.02) -> how far past near is now kept; is the following pause cut out again like main?
5. Superset property: at settings >= 400 ms, head's loud_frames ⊇ main's on >= 5000 random rms arrays (report violations). Below 400 ms, report differences and whether any lose speech in the sub-400 footsteps repro (the review accepted that case as a documented trade if the README notes it — check whether the README now notes that below 400 ms quiet sounds still join across 400 ms).
6. Breath / dense-breath / anchored fuzz still 0 pauses decoded or split; full test suite on head; head merged with main and with open PRs #79 (claude/great-brahmagupta-xxj0ky) and #80 (feat/quiet-forced-cuts) — full suite on the combination.
7. Nits: near-distance claim in the PR description; README "400 ms or more apart" wording.
Return: per item pass/fail with numbers, any new issue (severity under the bar: never cut or drop more speech than main in realistic cases), and a verdict merge / merge_with_followups / do_not_merge.`, { label: 'verify:#78 fixes', schema: {
  type: 'object',
  properties: {
    head: { type: 'string' }, tests: { type: 'string' },
    items: { type: 'string', description: 'per-item pass/fail with numbers' },
    newIssues: { type: 'array', items: { type: 'object', properties: { severity: { type: 'string' }, title: { type: 'string' }, detail: { type: 'string' } }, required: ['severity', 'title', 'detail'] } },
    verdict: { type: 'string', enum: ['merge', 'merge_with_followups', 'do_not_merge'] },
    why: { type: 'string' },
  },
  required: ['head', 'tests', 'items', 'newIssues', 'verdict', 'why'],
} })
