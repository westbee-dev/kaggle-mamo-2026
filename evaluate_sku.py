"""Вычисляет WAPE и MASE по каждому SKU для файла предсказаний.

Проверка на окне из train.csv: cutoff — последний месяц истории, фактом считаются следующие 3 месяца.
Последние 3 месяца train.csv, заканчивающегося 2021-11:
    python evaluate_sku.py --train train.csv --submission validation.csv --cutoff 2021-08
Окно из середины истории (факт: 2020-11, 2020-12, 2021-01):
    python evaluate_sku.py --train train.csv --submission validation.csv --cutoff 2020-10

Проверка по файлу с ответами (доступен только преподавателю):
    python evaluate_sku.py --train train.csv --submission submission.csv --solution solution.csv

Файл предсказаний содержит столбцы id и prediction, id имеет вид <sku>_<YYYY-MM>, например 0_2021-12.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd


SEASON_LENGTH = 12
HORIZON = 3


def split_history(train: pd.DataFrame, cutoff: str) -> tuple:
    """Отделяет HORIZON месяцев после cutoff в качестве фактических значений.

    Args:
        train: Таблица train.csv: столбец sku, столбец unit и по столбцу на месяц.
        cutoff: Последний месяц истории в формате YYYY-MM.

    Returns:
        Таблица с историей до cutoff включительно и таблица факта со столбцами id и actual.

    Raises:
        ValueError: Если cutoff не найден, после него меньше HORIZON месяцев или до него мало истории.
    """
    month_columns = _get_month_columns(train)
    cutoff_position = _find_cutoff_position(month_columns, cutoff)
    holdout_columns = month_columns[cutoff_position + 1:cutoff_position + 1 + HORIZON]
    history = train.drop(columns=month_columns[cutoff_position + 1:])
    melted = train.melt(id_vars='sku', value_vars=holdout_columns, var_name='month', value_name='actual')
    melted['id'] = melted['sku'].astype(str) + '_' + melted['month']
    return history, melted[['id', 'actual']]


def compute_wape(actuals: np.ndarray, predictions: np.ndarray) -> float:
    """Вычисляет WAPE по всем строкам.

    Args:
        actuals: Фактические значения.
        predictions: Предсказанные значения.

    Returns:
        Сумма модулей ошибок, делённая на сумму модулей факта.

    Raises:
        ValueError: Если сумма модулей факта равна нулю.
    """
    denominator = np.abs(actuals).sum()

    if denominator == 0:
        raise ValueError('Сумма модулей факта равна нулю, WAPE не определён.')

    return float(np.abs(actuals - predictions).sum() / denominator)


def compute_mase_per_sku(history: pd.DataFrame, evaluation: pd.DataFrame) -> pd.Series:
    """Вычисляет MASE каждого SKU с сезонным масштабом.

    Args:
        history: Таблица истории: столбец sku и по столбцу на месяц.
        evaluation: Таблица со столбцами id, actual и prediction.

    Returns:
        Значения MASE с индексом sku. SKU с нулевым масштабом в результат не входят.
    """
    values = history[_get_month_columns(history)].to_numpy(dtype=float)
    seasonal_differences = np.abs(values[:, SEASON_LENGTH:] - values[:, :-SEASON_LENGTH])
    scale = pd.Series(seasonal_differences.mean(axis=1), index=history['sku'].to_numpy())
    scale = scale[scale > 0]

    errors = evaluation.assign(
        sku=evaluation['id'].str.split('_').str[0].astype(int),
        error=(evaluation['actual'] - evaluation['prediction']).abs(),
    )
    mean_errors = errors.groupby('sku')['error'].mean()
    mase = mean_errors.reindex(scale.index) / scale
    return mase.rename('mase').rename_axis('sku')


def build_evaluation(actuals: pd.DataFrame, submission: pd.DataFrame) -> pd.DataFrame:
    """Соединяет факт и предсказания после проверки формата.

    Args:
        actuals: Таблица со столбцами id и actual.
        submission: Таблица со столбцами id и prediction.

    Returns:
        Таблица со столбцами id, actual и prediction.

    Raises:
        ValueError: Если строки не совпадают или предсказания некорректны.
    """
    if not {'id', 'prediction'}.issubset(submission.columns):
        raise ValueError('В файле предсказаний должны быть столбцы id и prediction.')

    if submission['id'].duplicated().any():
        raise ValueError('В столбце id файла предсказаний есть повторы.')

    if set(actuals['id']) != set(submission['id']):
        raise ValueError('Значения id в файле предсказаний не совпадают с оцениваемыми строками.')

    evaluation = actuals[['id', 'actual']].merge(submission[['id', 'prediction']], on='id')
    evaluation['prediction'] = pd.to_numeric(evaluation['prediction'], errors='coerce')

    if not np.isfinite(evaluation['prediction'].to_numpy(dtype=float)).all():
        raise ValueError('Столбец prediction должен содержать только конечные числа.')

    return evaluation


def _find_cutoff_position(month_columns: list, cutoff: str) -> int:
    """Находит позицию cutoff среди месяцев и проверяет, что истории и факта достаточно.

    Args:
        month_columns: Имена столбцов месяцев вида YYYY-MM в исходном порядке.
        cutoff: Последний месяц истории в формате YYYY-MM.

    Returns:
        Индекс месяца cutoff в списке month_columns.

    Raises:
        ValueError: Если cutoff не найден, после него меньше HORIZON месяцев или до него мало истории.
    """
    if cutoff not in month_columns:
        raise ValueError(f'Месяц {cutoff} не найден в train.csv.')

    cutoff_position = month_columns.index(cutoff)

    if len(month_columns) - cutoff_position - 1 < HORIZON:
        raise ValueError(f'После {cutoff} в train.csv меньше {HORIZON} месяцев для факта.')

    if cutoff_position + 1 < SEASON_LENGTH + 1:
        raise ValueError(f'До {cutoff} включительно меньше {SEASON_LENGTH + 1} месяцев истории для масштаба MASE.')

    return cutoff_position


def _get_month_columns(table: pd.DataFrame) -> list:
    """Возвращает столбцы месяцев в исходном порядке.

    Args:
        table: Таблица с историей продаж.

    Returns:
        Имена столбцов вида YYYY-MM.
    """
    return [column for column in table.columns if column[:2] == '20']


def _parse_arguments() -> argparse.Namespace:
    """Разбирает аргументы командной строки.

    Returns:
        Разобранные аргументы.
    """
    parser = argparse.ArgumentParser(description='Вычисляет WAPE и MASE по каждому SKU.')
    parser.add_argument('--train', type=Path, required=True, help='Путь к train.csv.')
    parser.add_argument('--submission', type=Path, required=True, help='Путь к файлу предсказаний.')
    parser.add_argument(
        '--cutoff',
        help='Последний месяц истории, YYYY-MM; следующие 3 месяца train.csv считаются фактом.',
    )
    parser.add_argument('--solution', type=Path, help='Путь к файлу с ответами.')
    parser.add_argument('--usage', choices=['Public', 'Private'], help='Часть файла с ответами для расчёта.')
    parser.add_argument('--output', type=Path, help='Путь для сохранения MASE по каждому SKU.')
    return parser.parse_args()


def main() -> None:
    """Выводит WAPE и сводку MASE по каждому SKU."""
    arguments = _parse_arguments()

    if (arguments.cutoff is None) == (arguments.solution is None):
        raise SystemExit('Укажите ровно один из аргументов: --cutoff или --solution.')

    if arguments.usage is not None and arguments.solution is None:
        raise SystemExit('Аргумент --usage работает только вместе с --solution.')

    train = pd.read_csv(arguments.train)

    if arguments.solution is not None:
        history, actuals = train, pd.read_csv(arguments.solution)
    else:
        try:
            history, actuals = split_history(train, arguments.cutoff)
        except ValueError as error:
            raise SystemExit(str(error))

    submission = pd.read_csv(arguments.submission)
    evaluation = build_evaluation(actuals, submission)

    if arguments.usage is not None:
        selected_ids = actuals.loc[actuals['Usage'] == arguments.usage, 'id']
        evaluation = evaluation[evaluation['id'].isin(selected_ids)]

    wape = compute_wape(evaluation['actual'].to_numpy(dtype=float), evaluation['prediction'].to_numpy(dtype=float))
    mase = compute_mase_per_sku(history, evaluation)

    print(f'WAPE: {wape:.4f}')
    print(f'Медиана MASE: {mase.median():.4f}')
    print(f'Доля SKU с MASE < 1: {(mase < 1).mean():.1%}')
    print(f'SKU в расчёте MASE: {len(mase)} из {len(history)}')

    if arguments.output is not None:
        mase.to_csv(arguments.output)


if __name__ == '__main__':
    main()
