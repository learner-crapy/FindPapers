SELECT link, title
FROM papers
WHERE TRUE = ANY(if_match);