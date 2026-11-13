/**
 * Capture validation (pure, Node-safe so scripts/parity_check.mjs can test it).
 */

/**
 * Character-level accuracy (%) between typed string and target prompt.
 */
export function computeTypingAccuracy(typed, target) {
  const a = String(typed || '');
  const b = String(target || '');
  if (!a.length) return 100;
  const len = Math.max(a.length, b.length);
  let matches = 0;
  const minLen = Math.min(a.length, b.length);
  for (let i = 0; i < minLen; i++) {
    if (a[i] === b[i]) matches++;
  }
  return Math.round((matches / len) * 100);
}

/**
 * `minLength` applies when a new password is enrolled (12 by default in model_config.json: longer
 * passwords carry more rhythm evidence, README §5.7); verification of an existing account passes 1.
 */
export function validatePasswordCapture(capture, expectedPassword, { minLength = 6 } = {}) {
  if (!expectedPassword || expectedPassword.length < minLength) {
    return {
      ok: false,
      code: 'SHORT_PASSWORD',
      message: `비밀번호는 최소 ${minLength}자 이상으로 설정해 주세요. 길수록 타이핑 리듬 근거가 많아져 다른 사람을 더 잘 구분합니다.`,
    };
  }
  if (capture.hadPaste) {
    return {
      ok: false,
      code: 'PASTE_BLOCKED',
      message: '붙여넣기는 사용할 수 없습니다. 직접 키보드로 타이핑해 주세요.',
    };
  }
  if (capture.typedText !== expectedPassword) {
    return {
      ok: false,
      code: 'PASSWORD_MISMATCH',
      message: '비밀번호 문자열이 일치하지 않습니다. 철자를 확인하고 다시 입력해 주세요.',
    };
  }
  if (capture.cleanDigraphCount < 4) {
    return {
      ok: false,
      code: 'INSUFFICIENT_CLEAN_KEYS',
      message: '유효한 연속 타건 구간이 너무 적습니다(과도한 수정). 한 번만 자연스럽게 다시 입력해 주세요.',
    };
  }
  return { ok: true };
}

export function validateSentenceCapture(capture, targetSentence) {
  if (capture.hadPaste) {
    return {
      ok: false,
      code: 'PASTE_BLOCKED',
      message: '붙여넣기는 사용할 수 없습니다. 타자연습 문장을 직접 타이핑해 주세요.',
    };
  }
  const typedTrim = capture.typedText.trim();
  const targetTrim = String(targetSentence || '').trim();
  const minChars = Math.max(8, Math.floor(targetTrim.length * 0.72));
  if (typedTrim.length < minChars) {
    return {
      ok: false,
      code: 'INCOMPLETE_SENTENCE',
      message: `문장을 끝까지 타이핑한 뒤 Enter를 눌러주세요 (${typedTrim.length} / ${targetTrim.length}자).`,
    };
  }
  if (capture.cleanDigraphCount < 8) {
    return {
      ok: false,
      code: 'INSUFFICIENT_CLEAN_KEYS',
      message: '추출된 연속 리듬 구간이 부족합니다. 문장을 자연스럽게 이어서 타이핑해 주세요.',
    };
  }
  return {
    ok: true,
    accuracy: computeTypingAccuracy(typedTrim, targetTrim),
  };
}
