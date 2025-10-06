SELECT link, title, query
FROM papers
WHERE TRUE = ANY(if_match);