# ExaModelsPEtab.jl

DO NOT MODIFY THIS FILE UNLESS EXPLICITLY ASKED.
IF ASKED, ALWAYS CONFIRM PROPOSED EDITS BEFORE MAKING CHANGES.
THIS FILE DEFINES WRITING/CODING/CONVERSATION CONVENTIONS FOR WORKING IN THIS PACKAGE.

## Conversation

- One step at a time: propose, the user confirms, then write.
- Never edit a comment the user wrote. Their explicit approval of a rename is the only exception.
  If a comment is TODO and you addressed it, write TODO (REVIEW) ... under the original TODO with
  a 1-2 sentence summary of what was done.

## Names and terminology

- Reuse names already in the files or the conversation.
- ALWAYS use plain definitions and simple clear language. Never invent coined terms that were not
  discussed beforehand, like anchor or stride. Never invent arbitrary notation that does not fit
  the rest of the codebase.
- Use straightforward language in plain English always: every other step, every 4th step, "fixed
  nodes" instead of "anchors".
- Axes are in the style of `cidx`, `ssidx`, `yidx`, `m`, `v`. Counts are `Nz`, `Nc`, `Nss`, `Ny`,
  `Ntheta_per_cond`. Index vectors are plural, `preeq_idxs`. e.g., `m = 1:Nm` for measurements,
  `cidx = 1:Nc` for conditions.

## Code text

- No spaces inside index brackets: `z0[v,cidx,i,k]`, `nodes[cidx][i+1]`.
- At most one terse comment per function stating what it does. DO NOT explain the details of what 
  the code is doing in the comment itself. The code should be clearly readable enough to where
  the user can clearly understand what the code is doing.
- For simple helper functions, no comment is fine.
- Indicies and definitions should be clear and easily readable. e.g, "for keyword in keywords"
  instead of "for kw in kws", etc.
