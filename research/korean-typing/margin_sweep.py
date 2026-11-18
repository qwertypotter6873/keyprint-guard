"""Measure the five-person touch margin curve with frozen similarities."""

import numpy as np

from metrics import METRICS, bootstrap, exact, pack


MARGINS = tuple(i / 100 for i in range(12, -1, -1))
THRESHOLD = 0.78


def aggregate(writer_outcomes):
    current = writer_outcomes[0]
    curve = []
    for gap, values in zip(MARGINS, writer_outcomes, strict=True):
        np.testing.assert_allclose(values[:, 1:4].sum(1), 1, atol=1e-12)
        np.testing.assert_allclose(values[:, 0], current[:, 0])
        np.testing.assert_allclose(values[:, 4], current[:, 4])
        delta = values - current
        curve.append({'margin': gap, 'rates': pack(values.mean(axis=0)),
                      'delta_from_0.12': pack(delta.mean(axis=0)),
                      'paired_writer_bootstrap_95_delta': bootstrap(delta)})
    for a, b in zip(writer_outcomes, writer_outcomes[1:]):
        assert np.all(b[:, [1, 2, 6]] >= a[:, [1, 2, 6]] - 1e-12)
    return curve


def all_rosters(answer, baseline):
    scores = np.asarray(answer['matrix'])
    n = len(scores)
    assert scores.shape == (n, n) and np.isfinite(scores).all()
    assert [r['writerIndex'] for r in answer['fixed']['genuine']] == list(range(n))
    values = [np.array([exact(row, i, THRESHOLD, gap) for i, row in enumerate(scores)]) for gap in MARGINS]
    curve = aggregate(values)
    assert curve[0]['rates'] == baseline['pr']['mean_over_all_five_person_rosters']
    assert curve[-1]['rates'] == baseline['pr_zero_margin_diagnostic']['mean_over_all_five_person_rosters']
    return curve


def fact_rosters(answer, n):
    known, unknown = answer['leads'][::2], answer['leads'][1::2]
    leads = np.asarray([(row['similarity'], row['lead']) for row in answer['leads']])
    assert leads.shape == (2 * n * 5, 2)
    assert len(known) == len(unknown) == n * 5
    win = np.array([row['winner'] for row in known]) == np.repeat(np.arange(n), 5)
    low = leads[::2, 0] < THRESHOLD
    rows = []
    for gap in MARGINS:
        yes = (~low) & (leads[::2, 1] >= gap)
        outsider = (leads[1::2, 0] >= THRESHOLD) & (leads[1::2, 1] >= gap)
        if gap == MARGINS[0]:
            assert list(yes) == [r['accepted'] for r in known]
            assert list(outsider) == [r['accepted'] for r in unknown]
        trials = np.column_stack((win, win & yes, ~win & yes, ~yes, low, ~yes & ~low, outsider))
        rows.append(trials.reshape(n, 5, len(METRICS)).mean(axis=1))
    curve = aggregate(rows)
    counts = answer['report']['candidate']
    deployed = curve[0]['rates']
    for metric, key in [('top1', 'top1_before_rejection'), ('accept_correct', 'accepted_correct'),
                        ('accept_wrong', 'accepted_wrong'), ('reject_known', 'rejected')]:
        assert deployed[metric] == counts['enrolled_probes'][key]['rate']
    assert deployed['accept_outsider'] == counts['unknown_writer_probes']['false_accept']['rate']
    return curve


def plot_validation(report, destination):
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import font_manager
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter

    primary, = [g for g in report['groups'] if g.get('split') == 'VS'
                and g['question_selection'] == 'chronological' and g['prefix_text_taps'] == 40]
    curve = primary['curve']
    margins = [v['margin'] for v in curve]
    korean_fonts = ['Apple SD Gothic Neo', 'NanumGothic', 'Noto Sans CJK KR', 'Malgun Gothic']
    installed = {font.name for font in font_manager.fontManager.ttflist}
    korean = any(name in installed for name in korean_fonts)
    labels = (['등록자를 실제 작성자로 수락', '등록자를 다른 사람으로 수락', '미등록자를 등록자로 수락'] if korean else
              ['Correct known acceptance', 'Wrong known acceptance', 'Outsider acceptance'])
    series = [('accept_correct', '#006c70', 'o', '-'),
              ('accept_wrong', '#66569b', 's', '--'),
              ('accept_outsider', '#b34029', '^', '-')]
    with plt.rc_context({'font.family': 'sans-serif', 'font.sans-serif': korean_fonts + ['DejaVu Sans'],
                         'axes.spines.top': False, 'axes.spines.right': False}):
        fig, ax = plt.subplots(figsize=(8.7, 5.1), layout='none')
        fig.subplots_adjust(left=.09, right=.96, bottom=.18, top=.72)
        title = '1·2위 점수 차이를 낮출 때의 수락률' if korean else 'Acceptance as the minimum score lead decreases'
        subtitle = ('AI Hub 한국어 검증 46명  ·  후보 5명  ·  1위 유사도 기준 0.78' if korean else
                    'AI Hub Korean validation, 46 writers, five candidates, minimum similarity 0.78')
        fig.text(.09, .945, title, fontsize=21, weight='bold', va='top')
        fig.text(.09, .86, subtitle, fontsize=12, color='#425366')
        for (metric, color, marker, style), label in zip(series, labels, strict=True):
            ax.plot(margins, [v['rates'][metric] for v in curve], label=label, color=color,
                    linewidth=2.5, marker=marker, markersize=4.6, linestyle=style)
        ax.legend(loc='lower left', bbox_to_anchor=(-.012, 1), ncol=3, frameon=False,
                  fontsize=10.5, handlelength=1.6, columnspacing=1.4)
        ax.set(xlim=(.122, -.002), ylim=(0, .87), yticks=np.arange(0, .81, .2))
        ticks = [.12,.10,.08,.06,.04,.02,0]
        ax.set_xticks(ticks, ['0.12 (현재)' if korean else '0.12 (deployed)', '0.10', '0.08', '0.06', '0.04', '0.02', '0.00'])
        ax.set_xlabel('최소 점수 차이' if korean else 'Minimum score lead', fontsize=12, labelpad=8)
        ax.yaxis.set_major_formatter(PercentFormatter(xmax=1, decimals=0))
        ax.set_axisbelow(True)
        ax.grid(axis='y', color='#dbe1e7', linewidth=.9)
        ax.spines[['left','bottom']].set_color('#aab4bc')
        ax.tick_params(axis='both', length=0, pad=6)
        note = ('등록자와 미등록자의 비율은 각 집단의 전체 시도를 기준으로 계산했습니다.' if korean else
                'Each known and outsider rate uses all probes in its own group.')
        fig.text(.09, .026, note, fontsize=10.2, color='#425366')
        fig.savefig(destination, dpi=175, facecolor='white')
        plt.close(fig)
